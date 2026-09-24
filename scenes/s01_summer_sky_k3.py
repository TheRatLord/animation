"""s01 helper (round 6): the hero cumulonimbus painted mass by mass with lib/clouds2 geometry.

A background painter's build-up instead of one lambert field: the tower is ~12 flat cauliflower masses
(clouds2.envelope + clouds2.cauliflower, detail only on the edge), painted back to front. Each mass:
  * its lit face is a near-flat warm-white / ivory plane (~92-95 %) with a gentle cool falloff toward
    its terminator (no sphere shading);
  * its shadow is a crescent tied to the mass's UNDERSIDE / away-from-sun side: the terminator is the
    mass's own silhouette pushed toward the sun (so it echoes the cauliflower scallops) -> a HARD edge on
    the sun side; the shadow then lightens (reflected sky) toward the far side -> soft far edge;
  * it throws a cool 'lip' onto the paint just behind its top edge (the tucked-in underside of the mass
    behind, ~225-232) -> each lit lobe gets a crisp upper edge against the mass it overlaps.
The shadow-side masses carry one coherent lavender body: lighter / warmer where they face the sky dome,
deeper blue-grey low down, with soft skylit crests on the right-hand lobes. The anvil is a wide thin
blade sheared downwind (right) with a cool shaded underside band and a torn lower edge; both ends fray
into fibres that stream into the cirrus. A 2-4 px silver lining runs on the sun-facing silhouette, and
the base tears into wisps and dissolves into the pale horizon haze."""
import math
import numpy as np
import cv2

from lib import clouds2 as K

_ss = K._ss


def _col(h):
    return np.asarray(K._c(h), np.float32)


HI = np.array([0.958, 0.948, 0.918], np.float32)     # warm ivory peak (~244)
LIT = np.array([0.935, 0.928, 0.905], np.float32)    # warm white (~238)
LIT_LO = np.array([0.895, 0.895, 0.918], np.float32)  # cool mid under the lip / near the terminator
TERM = np.array([0.83, 0.815, 0.875], np.float32)    # slightly warm-violet terminator band
SH_TOP = np.array([0.74, 0.765, 0.905], np.float32)  # shadow facing the sky dome (lighter, warmer)
SH_LOW = np.array([0.55, 0.61, 0.83], np.float32)   # shadow low in the tower (deeper blue-grey)
SH_CORE = np.array([0.62, 0.655, 0.86], np.float32)  # core shadow right at the terminator
REFL = np.array([0.76, 0.8, 0.93], np.float32)       # reflected light on the far side
SKY_CREST = np.array([0.8, 0.83, 0.95], np.float32)  # skylit crests of shadowed lobes
LIP = np.array([0.95, 0.955, 0.985], np.float32)     # multiplier: cool tuck behind a lobe's top edge
HAZE = _col('#d3e9f7')

LEV = ((0.16, 0.36, 0.95, 1.1, 2.2, 0.12, 0.45), (0.06, 0.14, 0.85, 1.0, 2.2, 0.3, 0.6),
       (0.028, 0.055, 0.5, 1.3, 3.0), (0.014, 0.022, 0.2, 1.8, 4.5))


def _dist_in(mb):
    return cv2.distanceTransform(mb.astype(np.uint8), cv2.DIST_L2, 5)


def _clean(b, min_area):
    """drop islands / fill holes of a boolean mask smaller than min_area px."""
    M = (b * 255).astype(np.uint8)
    K._clean(M, np.ones_like(b, bool), min_area)
    return M > 127


class Tower:
    def __init__(self, pw, ph, cx, base_y, Hc, W, sun, seed=12):
        self.u = W / 1920.0
        self.cx, self.by, self.Hc = cx, base_y, Hc
        self.sun = np.asarray(sun, np.float32)
        self.P = K.Painter(pw, ph, sun, 'noon', self.u, seed=seed + 1, sun_z=0.3)
        self.rng = np.random.default_rng(seed)
        self.pw, self.ph = pw, ph
        # far lighting key (upper left): per-mass direction for the shadow crescents
        self.key = np.array([cx - 1.1 * Hc, base_y - 2.0 * Hc], np.float32)

    # -------------------------------------------------------------- one painted mass
    def mass(self, dx, h0, w, hh, shade=0.35, lit=1.0, lip=1.0, crest=0.0, size=None, power=2.4, lump=0.1,
             lean=0.0, skew=0.0, base_round=0.2, levels=LEV, side=0.72, refl=0.55, hot=1.0, key=None,
             top_flat=0.0, down_cut=0.3, sun_bias=0.4):
        """dx/h0/w/hh in Hc units (h0 = bottom height above the base). shade = crescent depth (fraction
        of hh): 0.3 lit lobe with an underside shadow .. 2 = all shadow. lit - lit-face brightness 0..1."""
        Hc, u, P = self.Hc, self.u, self.P
        rng = self.rng
        cxp = self.cx + dx * Hc
        byp = self.by - h0 * Hc
        poly = K.envelope(cxp, byp, w * Hc, hh * Hc, rng, lean=lean, power=power, lump=lump,
                          base_round=base_round, skew=skew, top_flat=top_flat)
        size = (size or hh) * Hc
        kp = self.key if key is None else np.asarray(key, np.float32)
        sd = kp - np.array([cxp, byp - 0.5 * hh * Hc], np.float32)
        sd = sd / float(np.hypot(*sd))
        sx, sy = float(sd[0]), float(sd[1])
        m, x0, y0 = K.cauliflower(poly, np.random.default_rng(int(rng.integers(1 << 30))), size, levels,
                                  down_cut=down_cut, side_scale=side, sun_dir=(sx, sy), sun_bias=sun_bias,
                                  concave=0.7, clump=0.5)
        pad = int(0.25 * size + 8)
        m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0 -= pad
        y0 -= pad
        h, w_ = m.shape
        mb = m > 0.5
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w_].astype(np.float32)
        # ---- lit / shadow split: crescent on the away-from-sun side (terminator echoes the silhouette)
        D = shade * size
        lit_b = mb & (K._shift(m, -sx * D, -sy * D) > 0.5)
        # a little low-frequency wander so the terminator is not a pure offset copy of the outline
        if lit_b.any() and D > 0:
            tn = K._noise(w_, h, max(w_ / (0.35 * size), 2.0), int(rng.integers(1 << 20)), 3) - 0.5
            far = K._blur(K._shift(m, -sx * D, -sy * D), 0.06 * size)
            lit_b = mb & ((far + 0.5 * tn) > 0.55)
        lit_b = _clean(lit_b, (0.26 * size) ** 2)
        lit_b &= mb
        # crisp anti-aliased terminator; the lit value is carried over the silhouette's AA ring (else
        # every lit lobe gets a cool hairline outline where its AA pixels take the shadow colour)
        kd = 2 * int(math.ceil(1.5 * u)) + 1
        lit_d = cv2.dilate(lit_b.astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0
        lit_d = lit_d & ((m > 0.0) & ~(mb & ~lit_b))
        litf = K._blur(lit_d.astype(np.float32), 0.8 * u) * (m > 0.0)
        litf = np.maximum(litf, lit_d.astype(np.float32) * (~mb))
        # ---- lit face: near-flat ivory, cool falloff toward the terminator and the lower flank
        dT = cv2.distanceTransform((~lit_b).astype(np.uint8), cv2.DIST_L2, 5)    # 0 on lit
        shd_b = mb & ~lit_b
        # distance to the SHADOW (not to the silhouette): the terminator band never rings the outline
        dL = cv2.distanceTransform((~shd_b).astype(np.uint8), cv2.DIST_L2, 5) if shd_b.any() else             np.full(m.shape, 1e4, np.float32)
        # facing: blurred-mask gradient toward the sun (sun-facing edges brightest)
        mbl = K._blur(m, 0.12 * size)
        gx = cv2.Sobel(mbl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mbl, cv2.CV_32F, 0, 1, ksize=3)
        gl = np.sqrt(gx * gx + gy * gy) + 1e-6
        facing = np.clip(-(gx * sx + gy * sy) / gl, -1, 1) * _ss(0.0, 0.02, gl)
        # normalised blur (inside the mass only): the sun-facing value runs right up to the edge
        fm_ = mb.astype(np.float32)
        facing = K._blur((facing * fm_).astype(np.float32), 0.05 * size) / np.maximum(K._blur(fm_, 0.05 * size), 1e-3)
        vpos = np.clip((ys - (byp - hh * Hc)) / (hh * Hc), 0, 1)
        warm = np.clip(0.9 + 0.3 * facing - 0.7 * _ss(0.55, 1.0, vpos), 0, 1)
        # firm half-step: most of the face is one plane, the hot sun-facing top is a second plane
        step = K._blur(((facing - 0.35 * vpos) > 0.12).astype(np.float32), 1.0 * u)
        v = np.clip(0.7 * warm + 0.3 * step, 0, 1)
        cool = _ss(0.12 * size, 0.0, dL) * (1 - _ss(0.0, 1.0, D / size)) * 0.0
        lcol = LIT_LO[None, None] * (1 - v[..., None]) + LIT[None, None] * v[..., None]
        lcol = lcol + (HI - LIT)[None, None] * (hot * step * _ss(0.15, 0.7, facing))[..., None]
        # terminator band (lit side of the split): a narrow warm-violet mid value
        tb = _ss(0.07 * size, 0.0, dL) * lit_b
        tb = K._blur(tb.astype(np.float32), 1.2 * u) * 0.55
        lcol = lcol * (1 - tb[..., None]) + TERM * tb[..., None]
        if lit < 1.0:
            lcol = lcol * lit + LIT_LO * (1 - lit)
        # ---- shadow: tower-level value (lighter at the top of the body, deeper low down), hard at the
        # terminator, lightening toward the far side (reflected light)
        gy_ = _ss(self.by - 0.85 * Hc, self.by - 0.2 * Hc, ys)
        scol = SH_TOP[None, None] * (1 - gy_[..., None]) + SH_LOW[None, None] * gy_[..., None]
        if lit_b.any():
            core = np.exp(-dT / max(0.08 * size, 2.0))
        else:
            core = np.zeros_like(m)
        scol = scol * (1 - 0.5 * core[..., None]) + SH_CORE * (0.5 * core[..., None])
        # far side: toward the away-from-sun silhouette the shadow picks up reflected light (soft)
        dS = _dist_in(mb)
        away = np.clip((gx * sx + gy * sy) / gl, 0, 1) * _ss(0.0, 0.02, gl)
        rf = refl * _ss(0.3 * size, 0.0, dS) * K._blur(away.astype(np.float32), 0.1 * size)
        rf = K._blur(rf.astype(np.float32), 0.05 * size)
        scol = scol * (1 - rf[..., None]) + REFL * rf[..., None]
        if crest:
            # skylit crest: the up-facing top band of a shadowed mass catches the blue sky dome (soft)
            cb = m * (1 - K._shift_blur(m, 0.25 * sx * 0.06 * size, -0.06 * size, 0.02 * size))
            cb = K._blur(cb, 0.03 * size) * (1 - litf) * crest
            scol = scol * (1 - cb[..., None]) + SKY_CREST * cb[..., None]
        col = scol * (1 - litf[..., None]) + lcol * litf[..., None]
        # ---- paint texture (tiny broad value variation, not noise mush)
        tx = P._tex(P.tex, x0, y0, w_, h, xs, ys)
        col = col * (1 + 0.025 * tx[..., None])
        # ---- paste into the plate (with the lip on what is behind)
        px0, py0 = max(x0, 0), max(y0, 0)
        px1, py1 = min(x0 + w_, P.w), min(y0 + h, P.h)
        if px1 <= px0 or py1 <= py0:
            return
        sl = (slice(py0 - y0, py1 - y0), slice(px0 - x0, px1 - x0))
        reg = P.prem[py0:py1, px0:px1]
        if lip:
            dout = cv2.distanceTransform((~mb).astype(np.uint8), cv2.DIST_L2, 5)
            up = _ss(byp - 0.35 * hh * Hc, byp - 0.75 * hh * Hc, ys)       # only round the upper part
            lk = np.exp(-dout / max(0.045 * size, 2.0)) * (1 - m) * up * lip
            lk = lk[sl][..., None]
            mul = 1 - lk * (1 - LIP)
            reg[..., :3] *= mul
        a = m[sl][..., None]
        reg[..., :3] = reg[..., :3] * (1 - a) + col[sl] * a
        reg[..., 3:4] = reg[..., 3:4] * (1 - a) + a

    # -------------------------------------------------------------- anvil
    def anvil(self, xa, top, left, right, thick, seed=5):
        """Wide thin anvil blade sheared downwind (right): lit top, cool shaded underside band with a
        soft upper edge and a torn lower edge, both ends frayed into fibres."""
        P, u, Hc = self.P, self.u, self.Hc
        rng = np.random.default_rng(seed)
        # blade outline: domed over the updraft, thin, sheared downwind (the tip droops a little and
        # thins to a sliver), a short blunt upwind end that tears into fibres
        n = 160
        xx = np.linspace(xa - left, xa + right, n)
        uu = np.where(xx < xa, (xx - xa) / left, (xx - xa) / right)
        ar = np.abs(uu)
        ph1, ph2 = rng.uniform(0, 6.28, 2)
        thk = np.where(uu >= 0, thick * (1.0 - 0.85 * ar ** 0.8), thick * (1.0 - 0.6 * ar ** 1.6))
        dome = thick * 0.9 * np.exp(-(uu / 0.22) ** 2)
        top_ = top - dome + np.where(uu >= 0, thick * 0.5 * ar ** 1.5, thick * 0.25 * ar ** 2)             + thick * 0.12 * np.sin(ar * 9 + ph1) * ar
        bot_ = top_ + dome * 0.5 + thk + thick * 0.18 * np.sin(ar * 13 + ph2) * ar
        ap = np.concatenate([np.stack([xx, top_], 1), np.stack([xx[::-1], bot_[::-1]], 1)], 0).astype(np.float32)
        # shear: the downwind half droops a touch and the blade thins toward the tip
        pad = int(0.1 * Hc)
        X0 = int(ap[:, 0].min()) - pad
        Y0 = int(ap[:, 1].min()) - pad
        X1 = int(ap[:, 0].max()) + pad
        Y1 = int(ap[:, 1].max()) + pad
        w, h = X1 - X0, Y1 - Y0
        ss = 2
        M = np.zeros((h * ss, w * ss), np.uint8)
        cv2.fillPoly(M, [np.round((ap - [X0, Y0]) * ss).astype(np.int32)], 255, cv2.LINE_8)
        # soft low bumps on the top edge only (an anvil is flat and fibrous, not cauliflower)
        K._grow(M, np.random.default_rng(seed + 1), thick * ss * 1.2,
                ((0.05, 0.12, 0.5, 2.0, 4.0, 0.35, 0.6),), pref=(0.0, -1.0), down_cut=0.05, side_scale=0.2,
                clump=0.8, concave=0.8)
        m = K._resize(M.astype(np.float32) / 255.0, w, h, cv2.INTER_AREA)
        ys, xs = np.mgrid[Y0:Y0 + h, X0:X0 + w].astype(np.float32)
        # fibres: streak noise (horizontal) eats both ends
        st = K._noise(w, h, max(w / (0.05 * Hc), 3.0), seed + 3, 4, stretch=9.0)
        st2 = K._noise(w, h, max(w / (0.15 * Hc), 2.0), seed + 4, 3, stretch=5.0)
        fib = np.clip((0.7 * st + 0.3 * st2 - 0.5) * 2.4 + 0.5, 0, 1)
        endL = _ss(xa - 0.55 * left, xa - 1.02 * left, xs)
        endR = _ss(xa + 0.45 * right, xa + 1.0 * right, xs)
        # smear the tips outward (sheared fibres streaming out of the blade)
        src = m * np.maximum(endL, endR)
        acc = m.copy()
        for k in range(1, 17):
            f = (1 - k / 17.0) ** 0.7
            acc = np.maximum(acc, K._shift(src * (xs > xa), -0.03 * right * k / 16 * 6, 0) * f)
            acc = np.maximum(acc, K._shift(src * (xs < xa), 0.02 * left * k / 16 * 4, 0) * f)
        m = acc
        thr = np.maximum(endL * 0.7, endR * 0.9)
        a = m * _ss(thr - 0.12, thr + 0.12, fib * 0.85 + 0.15 * (1 - np.maximum(endL, endR)))
        # torn lower edge: streaks nibble the underside
        dn = m * (1 - K._shift(m, 0, 0.18 * thick))        # bottom band
        a = a * (1 - dn * _ss(0.55, 0.85, st2 * 0.5 + st * 0.5) * 0.85)
        a = K._blur(a, 0.6 * u)
        # values: lit top plane (warm near the tower, cooler downwind), cool underside band (soft top)
        band = K._blur(m * (1 - K._shift(m, 0, 0.5 * thick)), 0.16 * thick)
        band = np.clip(band * 1.3, 0, 1) * (1 - endL) * (1 - 0.5 * endR)   # frayed ends: backlit, bright
        down = _ss(xa, xa + right, xs)
        litc = LIT[None, None] * (1 - 0.4 * down[..., None]) + LIT_LO[None, None] * (0.4 * down[..., None])
        topb = m * (1 - K._shift(m, 0, -0.15 * thick))
        litc = litc + (HI - LIT) * K._blur(topb, 1.0 * u)[..., None] * (1 - down[..., None])
        shc = SH_TOP * 0.75 + REFL * 0.25
        col = litc * (1 - band[..., None]) + shc * band[..., None]
        # fibrous ice: faint streaks of value along the blade; it thins to translucency downwind
        col = col * (1 + 0.05 * (st[..., None] - 0.5))
        a = a * (1 - 0.5 * _ss(xa + 0.3 * right, xa + 1.0 * right, xs))
        tx = P._tex(P.tex, X0, Y0, w, h, xs, ys)
        col = col * (1 + 0.02 * tx[..., None])
        px0, py0 = max(X0, 0), max(Y0, 0)
        px1, py1 = min(X0 + w, P.w), min(Y0 + h, P.h)
        sl = (slice(py0 - Y0, py1 - Y0), slice(px0 - X0, px1 - X0))
        reg = P.prem[py0:py1, px0:px1]
        aa = a[sl][..., None]
        reg[..., :3] = reg[..., :3] * (1 - aa) + col[sl] * aa
        reg[..., 3:4] = reg[..., 3:4] * (1 - aa) + aa
        # long fibres continuing downwind into the cirrus, a few short ones upwind
        P.fibres(xa + 0.55 * right, xa + 1.45 * right, top + 0.55 * thick, 0.35 * thick, amount=0.6,
                 seed=seed + 7, lit=0.85)
        P.fibres(xa + 0.8 * right, xa + 1.7 * right, top + 0.9 * thick, 0.2 * thick, amount=0.4,
                 seed=seed + 9, lit=0.7)
        P.fibres(xa - 0.8 * left, xa - 1.35 * left, top + 0.45 * thick, 0.25 * thick, amount=0.45,
                 seed=seed + 11, lit=0.9)


def _silver_rim(rgba, sun, Hc, top, u):
    """Crisp 2-4 px silver-white lining just inside the sun-facing silhouette (per-pixel direction to
    the sun), strongest on the crown / upper-left flank next to the sun and only on lit paint; warmer
    (gold) right beside the sun. A faint halation spills a few px into the sky."""
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    h, w = A.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl

    def toward(img, r):
        return cv2.remap(img, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    r = 2.8 * u
    # isotropic ring (constant 2-3 px width whatever the edge angle to the sun)
    rb = max(int(round(r)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb + 1, 2 * rb + 1))
    core = K._blur(np.clip(A - cv2.erode(A, ker), 0, 1), 0.5 * u)
    soft = A * (1 - K._blur(toward(A, 2.5 * r), 0.8 * r))
    # only where the edge really faces the sun (not the sides of lobes grazing it)
    Ab = K._blur(A, 4 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    face = np.clip(-(gx * ux + gy * uy) / gl, 0, 1)
    face = _ss(0.15, 0.6, face)
    prox = np.exp(-dl / (0.75 * Hc))
    upper = 1 - _ss(top + 0.45 * Hc, top + 0.75 * Hc, ys)
    litn = _ss(0.8, 0.9, rgba[..., :3].max(-1))
    k = np.clip(core * 1.15 + 0.12 * soft, 0, 1) * face * (0.35 + 0.65 * prox) * (0.3 + 0.7 * upper) * litn
    # a cooler band just inside the lining (the lit face turning away under the edge) -> the lining pops
    band = np.clip(A - K._blur(cv2.erode(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(10 * u) + 1,) * 2)), 2.0 * u) - core, 0, 1)
    kb = (band * face * (0.35 + 0.65 * prox) * (0.3 + 0.7 * upper) * litn)[..., None]
    # never on thin fibres / wisps: only where the mass survives a morphological opening
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(7 * u) + 1,) * 2)
    solid = K._blur(cv2.dilate(cv2.erode((A > 0.5).astype(np.uint8), ko), ko, iterations=2).astype(np.float32), 1.5 * u)
    k = k * solid
    kb = kb * solid[..., None]
    k = np.clip(k, 0, 1)[..., None]
    near = np.exp(-dl / (0.2 * Hc))[..., None]
    col = np.array([1.45, 1.45, 1.42], np.float32) * (1 - near) + np.array([1.55, 1.4, 1.12], np.float32) * near
    out = rgba.copy()
    out[..., :3] = rgba[..., :3] * (1 - 0.07 * kb * np.array([1.0, 0.9, 0.55], np.float32))
    out[..., :3] = out[..., :3] * (1 - k) + col * k
    hk = (K._blur(k[..., 0] * A, 1.5 * u) * (1 - A) * 0.5)[..., None]
    a2 = np.clip(A[..., None] + hk * 0.5, 0, 1)
    out[..., :3] = (out[..., :3] * A[..., None] + col * hk * 0.5) / np.maximum(a2, 1e-4)
    out[..., :3] = np.where(a2 > 1e-4, out[..., :3], rgba[..., :3])
    out[..., 3:4] = a2
    return out


def tower(pw, ph, cx, base_y, Hc, W, hz_y, seed=12):
    """Paint the hero tower. Returns (rgba, sun_p)."""
    top = base_y - Hc
    sun = np.array([cx - 0.36 * Hc, top - 0.02 * Hc], np.float32)
    T = Tower(pw, ph, cx, base_y, Hc, W, sun, seed)
    u = T.u
    # ---------------- back: the shadowed body (one coherent lavender mass)
    T.mass(0.14, 0.02, 1.0, 0.46, shade=2.0, crest=0.0, lip=0.0, size=0.3, refl=0.35)          # base body
    T.mass(0.34, 0.14, 0.44, 0.42, shade=2.0, crest=0.6, lip=0.0, size=0.26, skew=0.2)          # right tower
    T.mass(0.1, 0.5, 0.44, 0.4, shade=2.0, crest=0.6, lip=0.0, size=0.3, lean=0.04)            # upper column
    T.mass(0.27, 0.46, 0.3, 0.26, shade=2.0, crest=0.7, lip=0.5, size=0.2, skew=0.2)            # right shoulder
    # ---------------- anvil (behind the crown, in front of the column top)
    T.anvil(cx + 0.02 * Hc, top + 0.085 * Hc, 0.34 * Hc, 1.0 * Hc, 0.07 * Hc, seed=seed + 40)
    # ---------------- crown: overshooting top punching through the anvil, next to the sun
    T.mass(-0.05, 0.84, 0.34, 0.17, shade=0.5, lip=1.0, size=0.16, power=2.2, base_round=0.3)
    T.mass(0.12, 0.82, 0.2, 0.1, shade=1.2, crest=0.5, lip=0.8, size=0.1, power=2.2, base_round=0.3)
    # ---------------- mid-body shadow-side masses (right), in front of the column
    T.mass(0.24, 0.3, 0.38, 0.3, shade=2.0, crest=0.35, lip=0.6, size=0.22, skew=0.15, refl=0.6)
    T.mass(0.4, 0.1, 0.34, 0.26, shade=2.0, crest=0.7, lip=0.5, size=0.2, refl=0.6)
    # ---------------- the lit face: three big ivory lobes stacked down the sunward flank (the upper one
    # bulges out under the crown, the middle one is tucked in, the lower one swells out again)
    # lit backing mass behind the three lobes: gaps between their florets show lit paint, not holes
    T.mass(-0.06, 0.22, 0.36, 0.58, shade=0.28, lip=0.0, size=0.3, power=2.8, base_round=0.3)
    T.mass(-0.1, 0.6, 0.42, 0.26, shade=0.3, lip=1.0, size=0.24, skew=-0.25, base_round=0.35)       # L1 (upper)
    T.mass(-0.11, 0.37, 0.4, 0.3, shade=0.3, lip=1.0, size=0.27, skew=-0.2, base_round=0.35)       # L2 (middle)
    T.mass(-0.17, 0.13, 0.56, 0.3, shade=0.45, lip=1.0, size=0.27, skew=-0.3, base_round=0.35)       # L3 (lower)
    # low front masses (shade side / base), catching a little skylight on top
    T.mass(0.17, 0.05, 0.42, 0.2, shade=2.0, crest=0.5, lip=0.8, size=0.18, refl=0.6)
    T.mass(-0.38, 0.02, 0.34, 0.18, shade=0.6, lip=0.8, size=0.16, base_round=0.3)
    P = T.P
    # ---------------- torn, wispy base dissolving into the horizon haze
    P.wisps(cx - 0.62 * Hc, cx + 0.7 * Hc, hz_y - 0.01 * Hc, 0.07 * Hc, rng=np.random.default_rng(seed + 3),
            seed=seed + 11, amount=0.5, erode=0.8, base_haze=0.5, haze_color=HAZE)
    yy = np.arange(P.h, dtype=np.float32)[:, None, None]
    P.prem *= _ss(hz_y + 0.06 * Hc, hz_y + 0.01 * Hc, yy)
    P.haze_band(hz_y - 0.22 * Hc, hz_y + 0.02 * Hc, HAZE, 0.6)
    # ---------------- silver lining on the sun-facing silhouette
    rgba = P.rgba()
    rgba = _silver_rim(rgba, sun, Hc, top, u)
    return rgba, sun
