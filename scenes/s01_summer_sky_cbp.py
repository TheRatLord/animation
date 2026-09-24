"""s01 helper (round 10): the hero cumulonimbus as a PAINTED LOBE STACK (clouds3 HeadSet geometry).

Why: the volumetric painter gave a noise-wobbled pillar with airbrushed interiors. A Shinkai
cumulonimbus is painted as discrete masses: every lobe is a flat, luminous lit plane with a crisp lit
top edge that follows its cauliflower scallops, a cool shade underneath, and the next (nearer, lower)
lobe's crisp lit edge overlapping that shade. So this module:

  * designs the massing by hand (units of the cloud height Hc): a broad heavy base (~1.3x the mid
    column), a narrowing column, a waist, then a wide cauliflower dome crown that overhangs the neck;
  * grows big heads -> medium heads -> florets on the exposed upper / outer arcs (children sit BEHIND
    their parent, so only their scalloped tops protrude and the parent's crisp lit edge cuts across
    their shade); every head carries clouds3.HeadSet cauliflower bumps (lumps + 3 scales of bumps);
  * paints each head back to front: a sun-side depth field (how far a pixel is from the head's
    outline toward the sun) makes the lit plane a crescent that follows the scalloped top; the
    terminator is crisp and brush-warped; the shade grades turn -> shade -> saturated cool core; the
    head's down-facing silhouette is lost (soft);
  * per-head exposure from the big form (sun top-left and behind the crown): the sunward flank is
    ~95 % luma white, the far flank / lower core is a saturated cobalt-violet, the crown in front of
    the sun turns luminous silver-lilac with a hot lining + transmitted glow on thin edge lobes;
  * a flat, darker, cooler base plane.
Everything is deterministic (seeded) and resolution independent (sizes scale with Hc / plate width).
"""
import math

import numpy as np
import cv2

from lib import clouds3 as K3

F32 = np.float32

PAL = dict(hi=(1.0, 0.995, 0.975), lit=(0.958, 0.958, 0.952), lit_lo=(0.86, 0.9, 0.96),
           turn=(0.73, 0.82, 0.95), shade=(0.5, 0.64, 0.92), deep=(0.35, 0.49, 0.89),
           base=(0.46, 0.56, 0.8), silver=(0.8, 0.86, 0.965), rim=(1.45, 1.36, 1.15), glow=(0.42, 0.36, 0.24))


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _c(v):
    return np.asarray(v, F32)


def _lerp(a, b, t):
    a, b = _c(a), _c(b)
    t = np.asarray(t, F32)
    return a + (b - a) * t[..., None] if t.ndim else a + (b - a) * float(t)


# ------------------------------------------------------------------------------------------ geometry
class Head:
    __slots__ = ('cx', 'cy', 'rx', 'ry', 'lvl', 'bumps', 'e', 'soft', 'flat_base', 'id')

    def __init__(self, cx, cy, rx, ry, lvl, bumps, e=0.6):
        self.cx, self.cy, self.rx, self.ry, self.lvl, self.bumps, self.e = cx, cy, rx, ry, lvl, bumps, e
        self.soft = 1.0
        self.flat_base = None


def _grow_bumps(rng, cx, cy, rx, ry, lvl, sc, sun_ang, fine=1.0):
    """clouds3.HeadSet cauliflower: lumps + edge bumps (+ 2 finer scales) on the upper arc."""
    hs = K3.HeadSet()
    if lvl == 0:
        kw = dict(bump=(0.06, 0.2), arc=(-205.0, 25.0), sub=0.6 * fine, lumps=(1, 3), lump_r=(0.3, 0.5),
                  sub3=0.4 * fine, spacing=1.15)
    elif lvl == 1:
        kw = dict(bump=(0.07, 0.22), arc=(-200.0, 20.0), sub=0.6 * fine, lumps=(0, 2), lump_r=(0.3, 0.5),
                  sub3=0.5 * fine, spacing=1.2)
    else:
        kw = dict(bump=(0.1, 0.28), arc=(-195.0, 15.0), sub=0.5 * fine, lumps=(0, 1), lump_r=(0.35, 0.5),
                  sub3=0.3 * fine, spacing=1.25)
    hs.add(cx, cy, rx, ry, rng, min_px=1.1 * sc, sun_ang=sun_ang, sun_bias=0.5, clump=0.65, **kw)
    B = np.asarray(hs.B, np.float64).reshape(-1, 4)
    return B[:, :3]


def hw_profile(s, prof):
    """Half width (units of Hc) at height fraction s by linear interpolation of prof [(s, hw)]."""
    ss = [p[0] for p in prof]
    hv = [p[1] for p in prof]
    return float(np.interp(s, ss, hv))


CB_PROF = ((0.0, 0.25), (0.15, 0.235), (0.3, 0.22), (0.45, 0.21), (0.6, 0.205), (0.72, 0.23), (0.8, 0.27),
           (0.88, 0.3), (0.94, 0.27), (0.98, 0.18), (1.0, 0.08))


def cb_layout(rng, cx, base_y, Hc, sun, sc, prof=CB_PROF, lean=0.05, fine=1.0, kids=1.0, dens=1.25, crown=True, smax=0.82, base_masses=True,
              towers=((-0.24, 0.4, 0.11, 0.075), (0.26, 0.27, 0.1, 0.07))):
    """Hand-designed cumulonimbus massing -> ordered list of heads (back -> front)."""
    ph = rng.uniform(0, 6.28, 3)
    sun_ang = math.atan2(sun[1] - (base_y - 0.9 * Hc), sun[0] - cx)

    def ax(s):
        return cx + Hc * (lean * s + 0.012 * math.sin(2 * math.pi * 1.3 * s + ph[0]))

    def expo(x, y):
        s = (base_y - y) / Hc
        hw = max(hw_profile(min(max(s, 0), 1), prof) * Hc, 1.0)
        xr = float(np.clip((x - ax(min(max(s, 0), 1))) / hw, -1.3, 1.3))
        e = 0.74 - 0.5 * xr + 0.22 * (s - 0.55)
        e -= 0.3 * _ss(0.3, 0.0, s)                 # heavy base in the shade of the mass above
        return float(np.clip(e, 0.03, 1.0))

    masses = []           # (sort key, x, y, rx, ry)
    if crown:
        # crown: one dominant dome + overhanging shoulders + a few turrets on top
        sC = 0.86
        hwC = hw_profile(sC, prof) * Hc
        masses.append((base_y - Hc, ax(sC) + 0.02 * Hc, base_y - sC * Hc, hwC * 0.9, hwC * 0.72))
        for side in (-1, 1):
            s_ = sC - rng.uniform(0.07, 0.1)
            x_ = ax(s_) + side * hwC * rng.uniform(0.55, 0.62)
            masses.append((base_y - s_ * Hc + 1, x_, base_y - s_ * Hc, hwC * rng.uniform(0.45, 0.52),
                           hwC * rng.uniform(0.36, 0.42)))
        for k in range(5):
            s_ = rng.uniform(0.9, 0.94)
            x_ = ax(s_) + rng.uniform(-0.7, 0.7) * hwC
            r_ = hwC * rng.uniform(0.3, 0.4)
            masses.append((base_y - Hc - 10 + k, x_, base_y - s_ * Hc + r_ * 0.6, r_, r_ * 0.78))
    # column: masses scattered through the envelope (power-law sizes, some bulging past the envelope on
    # the flanks), painted by the height of their top (upper = behind), so big lobes stack on big lobes
    area = sum(2 * hw_profile(s_, prof) * Hc * Hc / 40 for s_ in np.linspace(0, smax, 40))
    n0 = int(area / (math.pi * (0.13 * Hc) ** 2) * dens * 0.7) + 4
    for m in range(n0):
        s = smax * rng.random() ** 0.85
        hw = hw_profile(s, prof) * Hc
        r_ = hw * (0.3 + 0.38 * rng.random() ** 1.4)
        side = rng.uniform(-0.75, 0.75)
        x_ = ax(s) + side * max(hw - r_, 0.0)
        y_ = base_y - s * Hc + r_ * 0.35
        f = rng.uniform(0.84, 0.98)
        masses.append((y_ - r_ * f + rng.uniform(-0.6, 0.6) * r_, x_, y_, r_ * rng.uniform(1.0, 1.12), r_ * f))
    # flank masses: a clean envelope, staggered on each side (slightly behind the face)
    for sd_ in (-1, 1):
        s = rng.uniform(0.0, 0.06)
        while s < smax - 0.02:
            hw = hw_profile(s, prof) * Hc
            r_ = hw * rng.uniform(0.3, 0.46)
            x_ = ax(s) + sd_ * (hw - r_ * rng.uniform(0.35, 0.65))
            y_ = base_y - s * Hc + r_ * 0.4
            f = rng.uniform(0.82, 0.95)
            masses.append((y_ - r_ * f + rng.uniform(-0.3, 0.3) * r_, x_, y_, r_ * 1.05, r_ * f))
            s += r_ / Hc * rng.uniform(1.35, 1.8)
    # stepped sub-towers: (x offset, top height, base hw, top hw) in Hc
    for (xo, th, hb, ht) in towers:
        s = 0.0
        while s < th - 0.02:
            hw = (hb + (ht - hb) * s / th) * Hc
            r_ = hw * rng.uniform(0.85, 1.05)
            x_ = cx + xo * Hc + rng.uniform(-0.3, 0.3) * hw
            y_ = base_y - s * Hc + r_ * 0.3
            if base_y - th * Hc > y_ - r_ * 0.85:
                y_ = base_y - th * Hc + r_ * 0.85
            masses.append((y_ - r_ * 0.9 + rng.uniform(-0.2, 0.2) * r_, x_, y_, r_ * 1.08, r_ * 0.88))
            s += r_ / Hc * rng.uniform(0.9, 1.2)
    # spine: guarantees a filled column
    nsp = 4
    for m in range(nsp):
        s = (m + 0.5) / nsp * smax
        hw = hw_profile(s, prof) * Hc
        r_ = hw * rng.uniform(0.55, 0.65)
        y_ = base_y - s * Hc + r_ * 0.3
        masses.append((y_ - 0.2 * r_, ax(s) + rng.uniform(-0.2, 0.2) * hw, y_, r_ * 1.05, r_ * 0.92))
    # heavy flat base: two wide low masses in front, sitting on the base plane
    hw0 = hw_profile(0.0, prof) * Hc
    for side in ((-1, 1) if base_masses else ()):
        r_ = hw0 * 0.62
        masses.append((base_y + 1e5 + side, ax(0) + side * hw0 * 0.42, base_y - 0.05 * Hc, r_ * 1.15, r_ * 0.55))
    masses.sort(key=lambda m: m[0])

    heads = []

    def add(x, y, rx, ry, lvl):
        b = _grow_bumps(rng, x, y, rx, ry, lvl, sc, sun_ang, fine)
        h = Head(x, y, rx, ry, lvl, b, expo(x, y - 0.5 * ry))
        return h

    def kids_of(x, y, rx, ry, lvl, n_mean):
        """Children on the upper / outer arc; returned in paint order (deepest descendants first)."""
        out = []
        if lvl > 2:
            return out
        n = rng.poisson(n_mean * kids)
        s_ = (base_y - y) / Hc
        side = np.sign(x - ax(min(max(s_, 0), 1))) or 1.0
        for _ in range(n):
            a = math.radians(rng.uniform(-165, -15))
            if rng.random() < 0.55:
                a = 0.6 * a + 0.4 * (-math.pi / 2 + side * 0.9)
            r = 0.5 * (rx + ry) * rng.uniform(0.3, 0.48)
            if r < 3.0 * sc:
                continue
            ca, sa = math.cos(a), math.sin(a)
            rl = 1.0 / math.sqrt((ca / rx) ** 2 + (sa / ry) ** 2)
            k_ = rng.uniform(0.25, 0.55) if lvl == 0 else rng.uniform(0.4, 0.75)
            px = x + ca * (rl - r * k_)
            py = y + sa * (rl - r * k_)
            f = rng.uniform(0.72, 0.88)
            sub = kids_of(px, py, r * 1.05, r * f, lvl + 1, 1.3 if lvl == 0 else 0.8)
            out.extend(sub)
            out.append(add(px, py, r * 1.05, r * f, lvl + 1))
        return out

    for (_, x, y, rx, ry) in masses:
        s_ = (base_y - y) / Hc
        n_k = 5.0 if s_ > 0.72 else (2.2 if s_ > 0.5 else 1.5)
        for h in kids_of(x, y, rx, ry, 0, n_k):
            heads.append(h)
        heads.append(add(x, y, rx, ry, 0))
    return heads


# ------------------------------------------------------------------------------------------ painting
def _shift(B, dx, dy):
    """out[p] = B[p + (dx, dy)] (zero outside)."""
    h, w = B.shape
    out = np.zeros_like(B)
    ix, iy = int(round(dx)), int(round(dy))
    xs0, xs1 = max(0, -ix), min(w, w - ix)
    ys0, ys1 = max(0, -iy), min(h, h - iy)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0:ys1, xs0:xs1] = B[ys0 + iy:ys1 + iy, xs0 + ix:xs1 + ix]
    return out


def _head_mask(h, x0, y0, w, hh, ss=2):
    m = np.zeros((hh * ss, w * ss), np.uint8)
    S = 16 * ss

    def P(x, y):
        return int(round((x - x0) * S)), int(round((y - y0) * S))

    cv2.ellipse(m, P(h.cx, h.cy), (int(h.rx * S), int(h.ry * S)), 0, 0, 360, 255, -1, cv2.LINE_AA, 4)
    for (bx, by, br) in h.bumps:
        cv2.circle(m, P(bx, by), max(int(br * S), 1), 255, -1, cv2.LINE_AA, 4)
    return cv2.resize(m.astype(F32) / 255.0, (w, hh), interpolation=cv2.INTER_AREA)


def paint_heads_stack(heads, w, h, Hc, sun, sun_dir, base_y, pal=None, seed=0, near_sun=0.3, rim=1.0,
                      crest_up=0.3, T_lo=0.1, T_hi=1.05, shade_len=0.55, base_h=0.1, lost_px=None,
                      glow=1.0, silver=1.0, noise_amp=0.22):
    """Paint the ordered heads into a straight RGBA plate (h, w, 4)."""
    P_ = dict(PAL)
    P_.update(pal or {})
    sc = w / 1920.0
    col = np.zeros((h, w, 3), F32)
    A = np.zeros((h, w), F32)
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / np.linalg.norm(sd)
    # crescent direction: between the sun and straight up (the tops always catch the light)
    cd = np.array([sd[0] * (1 - crest_up), sd[1] * (1 - crest_up) - crest_up])
    cd /= np.linalg.norm(cd)
    rng = np.random.default_rng(seed)
    # global brush noise fields (plate space -> coherent across heads)
    nz_big = K3._noise(w // 4 + 1, h // 4 + 1, max(w / (70.0 * sc) / 4, 2), seed + 1, 3)
    nz_big = cv2.resize(nz_big, (w, h), interpolation=cv2.INTER_LINEAR)
    nz_fine = K3._noise(w // 2 + 1, h // 2 + 1, max(w / (16.0 * sc) / 2, 3), seed + 2, 2)
    nz_fine = cv2.resize(nz_fine, (w, h), interpolation=cv2.INTER_LINEAR)
    lp = lost_px if lost_px is not None else 0.008 * Hc
    for hd in heads:
        if len(hd.bumps):
            bx0 = min(hd.cx - hd.rx, float((hd.bumps[:, 0] - hd.bumps[:, 2]).min()))
            bx1 = max(hd.cx + hd.rx, float((hd.bumps[:, 0] + hd.bumps[:, 2]).max()))
            by0 = min(hd.cy - hd.ry, float((hd.bumps[:, 1] - hd.bumps[:, 2]).min()))
            by1 = max(hd.cy + hd.ry, float((hd.bumps[:, 1] + hd.bumps[:, 2]).max()))
        else:
            bx0, bx1, by0, by1 = hd.cx - hd.rx, hd.cx + hd.rx, hd.cy - hd.ry, hd.cy + hd.ry
        pad = int(lp * 2 + 4)
        x0, y0 = int(math.floor(bx0)) - pad, int(math.floor(by0)) - pad
        x1, y1 = int(math.ceil(bx1)) + pad, int(math.ceil(by1)) + pad
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        bw, bh = x1 - x0, y1 - y0
        M = _head_mask(hd, x0, y0, bw, bh)
        if base_y is not None:
            yy = np.arange(y0, y1, dtype=F32)[:, None]
            M = M * (1 - _ss(base_y - 0.004 * Hc, base_y + 0.002 * Hc, yy))
        B = M > 0.5
        R = 0.5 * (hd.rx + hd.ry)
        # sun-side depth: distance from the outline toward the (crest) light direction
        stp = max(0.75, R / 40.0)
        Tmax = 1.9 * R
        K = int(Tmax / stp) + 1
        ins = B.copy()
        d = np.zeros(B.shape, F32)
        for k in range(1, K + 1):
            ins &= _shift(B, cd[0] * k * stp, cd[1] * k * stp)
            if not ins.any():
                break
            d += ins * stp
        d = cv2.GaussianBlur(d, (0, 0), 0.6 + 0.02 * R)
        u = d / R
        # per-head params from its exposure in the big form
        e = hd.e
        dsun = math.hypot(hd.cx - sun[0], hd.cy - sun[1]) / Hc
        bl = math.exp(-(dsun / near_sun) ** 2) * silver          # backlit crown (right in front of the sun)
        T = (T_lo + (T_hi - T_lo) * e ** 1.3) * (1 - 0.85 * bl)
        sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        ps = (slice(cy0, cy1), slice(cx0, cx1))
        nb = nz_big[ps]
        nf = nz_fine[ps]
        u_ = u[sl] * (1 + noise_amp * nb) + 0.035 * nf
        us = cv2.GaussianBlur(u, (0, 0), 0.12 * R + 1)[sl]
        Mb = M[sl]
        e_px = 0.8 / R
        lit = 1 - _ss(T - e_px, T + e_px, u_)
        # lit plane: flat, a touch brighter right at the crisp top limb
        litc = _lerp(P_['lit_lo'], P_['lit'], min(1.0, 0.25 + 0.9 * e))
        litc = _lerp(_lerp(P_['turn'], P_['shade'], 0.25), litc, float(_ss(0.18, 0.62, e)))
        litc = _lerp(litc, P_['silver'], min(1.0, 1.1 * bl))
        limb = (1 - _ss(0.03, 0.09, u_)) * float(_ss(0.3, 0.8, e)) * (1 - bl)
        lc = litc + (_c(P_['hi']) - litc) * limb[..., None]
        # shade: turn -> shade -> core (cooler / more saturated for heads in the shadow of the big form)
        g = np.clip((us - T) / shade_len, 0, 1) ** 0.8 * (0.55 + 0.45 * e)
        sh_far = _lerp(P_['shade'], P_['deep'], float(np.clip(1.1 - 1.4 * e, 0, 1)))
        sh_far = _lerp(sh_far, P_['turn'], float(np.clip(1.6 * e - 0.6, 0, 1)))
        turn = _lerp(P_['turn'], P_['lit_lo'], 0.8 * e)
        turn = _lerp(turn, P_['shade'], float(np.clip(0.6 - 1.2 * e, 0, 0.6)))
        turn = _lerp(turn, P_['silver'], 0.5 * bl)
        sh_far = _lerp(sh_far, P_['turn'], 0.45 * bl)
        shc = turn[None, None, :] + (sh_far - turn)[None, None, :] * g[..., None]
        c = shc * (1 - lit[..., None]) + lc * lit[..., None]
        # lost silhouette on the down side: alpha fades in over lp px where the outline faces down
        Mbl = cv2.GaussianBlur(M, (0, 0), 2.0 + 0.1 * R)
        gy = cv2.Sobel(Mbl, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(Mbl, cv2.CV_32F, 1, 0, ksize=3)
        gl = np.sqrt(gx * gx + gy * gy) + 1e-6
        down = np.clip(-gy / gl, 0, 1)[sl]                      # outline facing down (gradient points up)
        dist = cv2.distanceTransform(B.astype(np.uint8), cv2.DIST_L2, 3).astype(F32)[sl]
        soft_a = _ss(0.0, lp * hd.soft, dist)
        dk = _ss(0.55, 0.9, down)
        a = Mb * (1 - dk + dk * soft_a)
        a = a[..., None]
        col[ps] = c * a + col[ps] * (1 - a)
        A[ps] = a[..., 0] + A[ps] * (1 - a[..., 0])
    # body fill behind the lobes: the gaps between lobes inside the silhouette are the cloud's shaded
    # body, never sky (closing of the union, painted UNDER the stack)
    kr = max(int(0.035 * Hc), 2)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
    Ab = (A > 0.5).astype(np.uint8)
    cl = cv2.morphologyEx(Ab, cv2.MORPH_CLOSE, ker).astype(F32)
    cl = cv2.GaussianBlur(cl, (0, 0), 1.0) * (1 - _ss(base_y - 0.004 * Hc, base_y + 0.002 * Hc,
                                                      np.arange(h, dtype=F32)[:, None]))
    xs_ = np.arange(w, dtype=F32)[None, :]
    lx = np.clip((xs_ - (sun[0] - 0.1 * Hc)) / (0.5 * Hc), 0, 1)
    body = _lerp(P_['turn'], P_['shade'], np.broadcast_to(lx, (h, w)))
    fill = np.clip(cl - A, 0, 1)[..., None]
    col = col + body * fill
    A = A + fill[..., 0]
    col = np.where(A[..., None] > 1e-4, col / np.maximum(A[..., None], 1e-4), 0)
    out = np.dstack([col, A]).astype(F32)
    return out


def finish(P, Hc, sun, sun_dir, base_y, pal=None, seed=0, rim=1.0, rim_reach=0.3, glow=1.0, base_h=0.15,
           kuwa=3, strokes=0.012, face_cool=0.12, fl=1.0):
    """Global passes on the stacked plate: flat cool base plane, hot lining + transmitted glow near the
    sun, cooler camera face under the crown, a light Kuwahara brush pass, fine silhouette florets."""
    P_ = dict(PAL)
    P_.update(pal or {})
    h, w = P.shape[:2]
    sc = w / 1920.0
    col, A = P[..., :3].copy(), P[..., 3].copy()
    nzr = np.nonzero(A.max(1) > 0.01)[0]
    nzc = np.nonzero(A.max(0) > 0.01)[0]
    y0, y1 = max(nzr.min() - 20, 0), min(nzr.max() + 20, h)
    x0, x1 = max(nzc.min() - 20, 0), min(nzc.max() + 20, w)
    c = col[y0:y1, x0:x1]
    a = A[y0:y1, x0:x1]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    if kuwa:
        c = K3.kuwahara(c, max(1, int(round(kuwa * sc))), q=6.0)
    # base plane: flat, darker, cooler
    if base_y is not None:
        bd = _ss(base_y - base_h * Hc, base_y, ys)
        c = c + (_c(P_['base']) - c) * (0.85 * bd ** 1.6)[..., None]
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    ins = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    # camera face beneath the crown grades cooler (the light comes from behind)
    if face_cool:
        fc = np.exp(-((dsun - 0.2) / 0.22) ** 2) * _ss(0.01 * Hc, 0.06 * Hc, ins) * (ys > sun[1])
        c = c + (_c((0.78, 0.83, 0.97)) - c) * (face_cool * fc)[..., None]
    # hot lining along the sun-facing silhouette near the sun + glow transmitted through thin edge lobes
    if rim:
        Ab = cv2.GaussianBlur(a, (0, 0), 3.0 * sc + 1)
        gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
        gl = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = -gx / gl, -gy / gl
        to_x, to_y = sun[0] - xs, sun[1] - ys
        tl = np.sqrt(to_x ** 2 + to_y ** 2) + 1e-6
        face = np.clip((nx * to_x + ny * to_y) / tl, 0, 1)
        up = np.clip(-ny, 0, 1)
        near = np.exp(-(dsun / rim_reach) ** 2)
        rw_ = 2.4 * sc + 0.6
        band = 1 - _ss(rw_ * 0.75, rw_ * 1.25, ins)
        k = band * np.clip(near * face ** 1.5 + 0.2 * up * face * np.exp(-dsun / 0.8), 0, 1) * rim
        c = c + (_c(P_['rim']) - c) * np.clip(k, 0, 1)[..., None]
        gl_ = np.exp(-ins / (0.005 * Hc)) * np.exp(-(dsun / (rim_reach * 0.55)) ** 2) * face * glow
        c = c + _c(P_['glow']) * gl_[..., None]
    # bright limb: the sky-facing silhouette of the lit masses is the whitest paint on the cloud
    Ab2 = cv2.GaussianBlur(a, (0, 0), 3.0 * sc + 1)
    gy2 = cv2.Sobel(Ab2, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(Ab2, cv2.CV_32F, 1, 0, ksize=3)
    gl2 = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-6
    upf = np.clip((gy2 * 0.8 + gx2 * 0.6) / gl2, 0, 1)             # faces up / toward the upper-left sun
    lum = c.mean(-1)
    lw_ = (3.0 + 1.5 * K3._noise(x1 - x0, y1 - y0, max((x1 - x0) / (40.0 * sc), 3), seed + 9, 2)) * sc + 0.5
    lb = (1 - _ss(lw_ * 0.8, lw_ * 1.15, ins)) * upf * _ss(0.7, 0.9, lum)
    c = c + (_c(P_['hi']) * 1.03 - c) * (0.7 * lb)[..., None]
    if strokes:
        st = K3._noise(x1 - x0, y1 - y0, max((x1 - x0) / (10.0 * sc), 6), seed + 41, 2, stretch=5.0, angle=-30)
        c = c * (1 + strokes * st[..., None])
    col[y0:y1, x0:x1] = c
    out = np.dstack([col, A]).astype(F32)
    if fl:
        K3.edge_florets(out, sun_dir=sun_dir, r=(1.5, 9.0), density=1.5 * fl, up_min=-0.1, sun_w=0.5, seed=seed,
                        lift=0.03)
        K3.edge_florets(out, sun_dir=sun_dir, r=(1.0, 3.5), density=1.2 * fl, up_min=0.1, sun_w=0.5, seed=seed + 1,
                        lift=0.03)
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


def hero(pw, ph, cx, base_y, Hc, W, hz_y, seed=7, sun_dir=(-0.8, -0.6), **kw):
    """Returns (rgba plate (ph, pw, 4), sun position in plate px)."""
    sc = pw / 1920.0
    rng = np.random.default_rng(seed)
    # sun: just behind the crown's upper-left shoulder
    sun = np.array([cx - 0.06 * Hc, base_y - 0.97 * Hc], np.float32)
    heads = cb_layout(rng, cx, base_y, Hc, sun, sc, **kw.get('layout', {}))
    P = paint_heads_stack(heads, pw, ph, Hc, sun, sun_dir, base_y, seed=seed, **kw.get('paint', {}))
    # put the sun right on the crown's silhouette (partly hidden): column x, just below the top edge
    col_ = np.nonzero(P[:, int(sun[0]), 3] > 0.5)[0]
    if len(col_):
        sun[1] = col_[0] + 0.03 * Hc
    P = finish(P, Hc, sun, sun_dir, base_y, seed=seed, **kw.get('finish', {}))
    return P, sun


HEAP_PROF = ((0.0, 0.92), (0.25, 1.0), (0.55, 0.86), (0.8, 0.56), (0.93, 0.3), (1.0, 0.1))


def family_plate(pw, ph, clouds, sun, sun_dir=(-0.8, -0.6), haze=0.2, haze_col=(0.8, 0.88, 0.98), seed=0):
    """Secondary cumulus painted with the same lobe stack. clouds: dicts(cx, base_y, width, height, seed,
    [haze]) listed far -> near. Returns a straight RGBA plate."""
    sc = pw / 1920.0
    acc = np.zeros((ph, pw, 3), F32)
    A = np.zeros((ph, pw), F32)
    for c in clouds:
        Hc = c['height']
        k = c['width'] / 2.0 / Hc
        prof = tuple((s_, v * k) for s_, v in HEAP_PROF)
        rng = np.random.default_rng(c.get('seed', 0))
        heads = cb_layout(rng, c['cx'], c['base_y'], Hc, sun, sc, prof=prof, lean=c.get('lean', 0.03), crown=False,
                          smax=0.8, base_masses=False, dens=c.get('dens', 1.1), kids=0.8,
                          towers=())
        P = paint_heads_stack(heads, pw, ph, Hc, sun, sun_dir, c['base_y'], seed=c.get('seed', 0), silver=0.0,
                              lost_px=0.03 * Hc)
        P = finish(P, Hc, sun, sun_dir, c['base_y'], seed=c.get('seed', 0), rim=0.4, glow=0.0, face_cool=0.0,
                   base_h=0.18, fl=0.6)
        hz = c.get('haze', haze)
        rgb = P[..., :3] + (np.asarray(haze_col, F32) - P[..., :3]) * hz
        a = P[..., 3:4]
        acc = rgb * a + acc * (1 - a)
        A = a[..., 0] + A * (1 - a[..., 0])
    rgb = np.where(A[..., None] > 1e-4, acc / np.maximum(A[..., None], 1e-4), 0)
    rgb = K3._bleed(rgb.astype(F32), (A > 0.02).astype(F32), max(3.0, 0.004 * pw))
    return np.dstack([rgb, A]).astype(F32)


def mackerel_rows(pw, ph, box, seed=0, angle=-11.0, rows=16, curve=0.00012, avoid=(), opacity=0.8,
                  size=(0.011, 0.0045), sc=1.0):
    """Altocumulus / mackerel sky as ALIGNED RIPPLED ROWS: parallel wavy rows following a gentle perspective
    curve (rows bunch up and shrink toward the far / lower edge of the box), each row a string of small
    elongated cloudlets separated by gaps; soft lit top edges, lost undersides, gathered in patches.
    box = (x0, y0, x1, y1) plate px; avoid = [(x, y, rx, ry)] soft holes. Returns straight RGBA."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    ss = 2
    m = np.zeros((ph * ss, pw * ss), np.uint8)
    S = 16 * ss
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    xc = 0.5 * (x0 + x1)
    L = (x1 - x0) * 1.3
    for r in range(rows):
        f = r / max(rows - 1, 1)                       # 0 near (top) .. 1 far (bottom)
        yy = y0 + (y1 - y0) * (1 - (1 - f) ** 1.35)    # rows compress toward the far edge
        scale = 1.0 - 0.6 * f
        hgt = size[1] * pw * scale
        ph_ = rng.uniform(0, 6.28)
        u = -L / 2 + rng.uniform(0, 1) * size[0] * pw
        while u < L / 2:
            wd = size[0] * pw * scale * rng.uniform(0.35, 1.6) ** 1.2
            ripple = math.sin(u / (0.05 * pw) + ph_) * hgt * 0.6
            vy = yy + curve * u * u + ripple
            px = xc + u * ca
            py = vy + u * sa + rng.normal() * hgt * 0.35
            h_ = hgt * rng.uniform(0.6, 1.2)
            ang = angle + rng.uniform(-8, 8)
            cv2.ellipse(m, (int(px * S), int(py * S)), (max(int(wd / 2 * S), 1), max(int(h_ / 2 * S), 1)), ang, 0, 360,
                        255, -1, cv2.LINE_AA, 4)
            # a small companion lump on top (lumpy, not a pill)
            if rng.random() < 0.6:
                cv2.ellipse(m, (int((px + rng.uniform(-0.25, 0.25) * wd) * S), int((py - h_ * 0.35) * S)),
                            (max(int(wd * 0.25 * S), 1), max(int(h_ * 0.4 * S), 1)), ang, 0, 360, 255, -1, cv2.LINE_AA, 4)
            u += wd * rng.uniform(0.9, 1.5) + (rng.random() < 0.12) * rng.uniform(1, 4) * wd
    M = cv2.resize(m.astype(F32) / 255.0, (pw, ph), interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(M, (0, 0), 1.5 * sc + 0.4)
    # patches: the field gathers into drifting clumps, strongest in the middle of the box
    pn = K3._noise(pw // 8 + 1, ph // 8 + 1, max(pw / 8 / (0.12 * pw / 8), 2), seed + 3, 3)
    pn = cv2.resize(pn, (pw, ph), interpolation=cv2.INTER_LINEAR)
    patch = _ss(-0.05, 0.45, pn)
    ys = np.arange(ph, dtype=F32)[:, None]
    xs = np.arange(pw, dtype=F32)[None, :]
    bx = _ss(x0, x0 + 0.15 * (x1 - x0), xs) * _ss(y0 - 0.02 * ph, y0 + 0.1 * (y1 - y0), ys) *         (1 - _ss(y1 - 0.2 * (y1 - y0), y1, ys))
    for (ax, ay, rx, ry) in avoid:
        bx = bx * _ss(0.7, 1.3, np.sqrt(((xs - ax) / rx) ** 2 + ((ys - ay) / ry) ** 2))
    lit = np.clip(soft - _shift(soft, 0, -max(1.5 * sc, 1.0)), 0, 1)       # top edge
    under = np.clip(soft - _shift(soft, 0, max(1.5 * sc, 1.0)), 0, 1)      # bottom edge
    col = np.zeros((ph, pw, 3), F32) + _c((0.88, 0.93, 0.995))
    col = col + (_c((1.03, 1.02, 1.0)) - col) * np.clip(lit * 2.5, 0, 1)[..., None]
    col = col + (_c((0.7, 0.8, 0.95)) - col) * np.clip(under * 1.5, 0, 0.6)[..., None]
    a = np.clip(soft - 0.4 * under, 0, 1) * patch * bx * opacity * (0.45 + 0.55 * (1 - (ys - y0) / max(y1 - y0, 1)))
    a = np.clip(a, 0, 1)
    col = K3._bleed(col, (a > 0.01).astype(F32), 3.0)
    return np.dstack([col, a]).astype(F32)


if __name__ == '__main__':
    import sys
    import os
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import sky as S, core as C
    W, H = (1920, 1080) if 'full' in sys.argv else (960, 540)
    pw, ph = int(W * 1.12), int(H * 1.3)
    t0 = time.time()
    base_y = ph - 0.25 * H
    Hc = 0.76 * H
    P, sun = hero(pw, ph, pw * 0.42, base_y, Hc, W, base_y)
    print('hero', time.time() - t0)
    sky = S.sky_gradient(pw, ph, dict(stops=[(0.0, '#0a45b8'), (0.3, '#2e80de'), (0.6, '#8ec4f0'), (1.0, '#ebf8fc')],
                                      sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc'), horizon=0.95)[..., :3]
    img = sky * (1 - P[..., 3:4]) + P[..., :3] * P[..., 3:4]
    C.save_png(os.path.join('out', 'compare', 'cbp_test.png'), np.clip(img, 0, 1))
