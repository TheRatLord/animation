"""City renderer for s03_city_dusk: a small software rasterizer of axis-aligned 3D boxes (buildings, setbacks,
rooftop clutter, fences) seen through a pinhole camera (no pitch), with procedurally painted facades.

Every face is an axis-aligned plane, so for each pixel we recover exact world coordinates on the face
and evaluate the facade pattern (floors, bays, windows, fins, spandrels, signs) analytically,
box-filtered by the pixel footprint (no moire on tiny far windows). Output per depth band:
  rgb (reflected / ambient colour), alpha, emission rgb (lights), emission-weighted switch-on time.

World: x right, y up (height), z forward (away from camera). Camera at (cam_x=0, CAM_H, 0).
Screen: sx = cx + f * x / z,  sy = hy + f * (CAM_H - y) / z.

Lighting design (painted, not physical): the sun has just set straight ahead-left. Faces toward the
camera (east) get a cool teal-blue sky ambient; faces toward the afterglow get warm light and mirror the
orange -> pink -> violet sky on glass, with a hot specular sliver on the sunward vertical corner; roofs
reflect the violet zenith. Near the ground everything glows with warm sodium street light.
"""
import math
import numpy as np
from numba import njit, prange

# building record columns
(BX0, BX1, BZ0, BZ1, BH, MAT, SEED, AR, AG, AB, FLH, BAYW, WWF, WHF, LIT0, LITADD, LITCOL, GLASS, BASE,
 GRP, SIGN, RUNF, ROOFT, CROWN, PAL) = range(25)
NB = 25
R_TX = 24.0     # texels per text line / char cell in the text atlases (s03_city_dusk_text.R)

# materials
M_CONC, M_GLASS, M_APT, M_BILL, M_FINS, M_BAND, M_DARK, M_FENCE, M_PLAIN, M_TANK, M_CURTAIN, M_NAME = range(12)


@njit(cache=True, inline='always')
def _hash(a, b, c, d):
    h = (a * 374761393 + b * 668265263 + c * 2246822519 + d * 3266489917) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(cache=True, inline='always')
def _pulse_int(x, period, a, w):
    n = math.floor(x / period)
    r = x - n * period
    return n * w + min(max(r - a, 0.0), w)


@njit(cache=True, inline='always')
def _pulse(x, fw, period, a, w):
    """box-filtered periodic pulse (filter width fw), 0..1"""
    if fw < 1e-6:
        r = x - math.floor(x / period) * period
        return 1.0 if (r >= a and r < a + w) else 0.0
    return (_pulse_int(x + 0.5 * fw, period, a, w) - _pulse_int(x - 0.5 * fw, period, a, w)) / fw


@njit(cache=True, inline='always')
def _sstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline='always')
def _lerp3(a0, a1, a2, b0, b1, b2, t):
    return a0 + (b0 - a0) * t, a1 + (b1 - a1) * t, a2 + (b2 - a2) * t


@njit(cache=True, inline='always')
def _sky_mirror(v):
    """colour of the afterglow sky mirrored in a sunward glass face at height v (m)"""
    k = min(max(v / 240.0, 0.0), 1.0)
    if k < 0.45:
        return _lerp3(1.35, 0.72, 0.4, 1.15, 0.48, 0.55, k / 0.45)
    return _lerp3(1.15, 0.48, 0.55, 0.5, 0.36, 0.78, (k - 0.45) / 0.55)


@njit(cache=True, inline='always')
def _glass_refl(u, vb, seed, fl, ci):
    """Round 6: painted reflection of the cumulus / cloud streaks in a curtain wall - big shapes with a
    firm edge, broken per mirror panel (each 2-bay x 3-floor panel is tilted a little, so the reflection
    steps at the panel joints like real glass). Returns the reflection mask 0..1."""
    pj = (_hash(seed, int(fl) // 3, int(ci) // 2, 61) - 0.5) * 4.0
    vv = vb + pj
    # long diagonal streaks (reflected cloud streaks / cumulus flank), only in the upper part of the tower
    c = 0.5 + 0.32 * math.sin(u * 0.035 - vv * 0.055 + seed * 0.37) + 0.18 * math.sin(u * 0.012 + vv * 0.01 + seed)         + 0.08 * math.sin(u * 0.11 - vv * 0.16 + seed * 2.3)
    up = _sstep(25.0, 120.0, vb)
    # round 7: a second, much larger reflected cumulus mass (big soft-edged shape across many panels)
    c2 = 0.5 + 0.35 * math.sin(u * 0.016 + vv * 0.021 + seed * 1.7) + 0.2 * math.sin(u * 0.041 - vv * 0.013 + seed)
    big = _sstep(0.72, 0.8, c2) * 0.5
    return max(_sstep(0.66, 0.72, c) * up, big * _sstep(20.0, 90.0, vb)) *         (0.75 + 0.25 * _hash(seed, int(fl) // 3, int(ci) // 2, 62))


@njit(cache=True)
def _fence_gap(u, v, hb, fu):
    """True where a fence / lattice face is see-through (posts every 1.8 m, top + mid rail)."""
    if v > hb - 0.22 or v < 0.15:
        return False
    if abs(v - hb * 0.5) < 0.1:
        return False
    r = u - math.floor(u / 1.8) * 1.8
    if r < 0.16 + 0.5 * fu:
        return False
    # diagonal bracing
    d = (u + v) - math.floor((u + v) / 1.8) * 1.8
    if d < 0.08 + 0.35 * fu:
        return False
    return True


@njit(cache=True, inline='always')
def _bil(A, l, e, y, x):
    h, w = A.shape[2], A.shape[3]
    y = min(max(y - 0.5, 0.0), h - 1.001)
    x = min(max(x - 0.5, 0.0), w - 1.001)
    y0, x0 = int(y), int(x)
    fy, fx = y - y0, x - x0
    return ((A[l, e, y0, x0] * (1 - fx) + A[l, e, y0, x0 + 1] * fx) * (1 - fy) +
            (A[l, e, y0 + 1, x0] * (1 - fx) + A[l, e, y0 + 1, x0 + 1] * fx) * fy)


@njit(cache=True, inline='always')
def _tex(A, e, y, x, ratio):
    """trilinear sample of a real-text coverage atlas A[level, entry, y, x] (texel coords); ratio = texels per
    sample footprint -> mip level. Outside the atlas -> 0."""
    if y < 0.0 or x < 0.0 or y > A.shape[2] or x > A.shape[3]:
        return 0.0
    lev = math.log2(max(ratio, 1.0))
    lev = min(lev, A.shape[0] - 1.0)
    l0 = int(lev)
    fr = lev - l0
    l1 = min(l0 + 1, A.shape[0] - 1)
    return _bil(A, l0, e, y, x) * (1 - fr) + _bil(A, l1, e, y, x) * fr


@njit(cache=True)
def _facade(u, v, fu, fv, B, face, z, out, HA, HN, NA, NN):
    """Facade at face coords u (horizontal m), v (height above base m), footprint fu, fv (m).
    face: 0 front (toward camera), 1 side sunward (faces -x), 2 side away (+x).
    writes out[0:3] base colour, out[3:6] emission, out[6] on-time, out[7] emission weight"""
    mat = int(B[MAT])
    seed = int(B[SEED])
    hb = B[BH] - B[BASE]
    ar, ag, ab = B[AR], B[AG], B[AB]
    out[3], out[4], out[5], out[6], out[7] = 0.0, 0.0, 0.0, 0.0, 0.0
    hv = min(max((v + B[BASE]) / 200.0, 0.0), 1.0)
    # ---- ambient light per face
    if face == 0:      # facing camera (east): cool teal-blue sky ambient
        lr, lg, lb = 0.12 + 0.05 * hv, 0.14 + 0.06 * hv, 0.24 + 0.08 * hv
    elif face == 1:    # facing the afterglow: warm
        lr, lg, lb = 0.40 + 0.2 * hv, 0.24 + 0.08 * hv, 0.28 + 0.03 * hv
    else:              # facing away: deep blue
        lr, lg, lb = 0.05, 0.065, 0.16
    # ground occlusion near the base, sodium street glow washing up the lowest floors
    vb = v + B[BASE]
    ao = (0.62 + 0.38 * _sstep(0.0, 14.0, vb))
    wr, wg, wb = lr * ar * ao, lg * ag * ao, lb * ab * ao
    sg = math.exp(-vb / 7.0) * 0.3
    wr += sg * 1.0
    wg += sg * 0.52
    wb += sg * 0.2
    if mat == M_BILL:
        # rooftop billboard panel: printed ad (unlit ones reflect the sky, lit ones glow)
        hb3 = hb
        if face != 0 or v < 1.2 or v > hb3 - 0.2 or u < 0.2 or u > B[BX1] - B[BX0] - 0.2:
            out[0], out[1], out[2] = wr * 0.5, wg * 0.5, wb * 0.6
            return
        k = seed % 6
        pw_ = B[BX1] - B[BX0] - 0.4
        uu = (u - 0.2) / max(pw_, 0.1)
        vv = (v - 1.2) / max(hb3 - 1.4, 0.1)
        if k == 0:
            br, bg, bb2, ar2, ag2, ab2 = 0.95, 0.95, 0.9, 0.85, 0.12, 0.15
        elif k == 1:
            br, bg, bb2, ar2, ag2, ab2 = 0.15, 0.35, 0.8, 1.0, 1.0, 1.0
        elif k == 2:
            br, bg, bb2, ar2, ag2, ab2 = 1.0, 0.78, 0.2, 0.2, 0.15, 0.3
        elif k == 3:
            br, bg, bb2, ar2, ag2, ab2 = 0.9, 0.2, 0.25, 1.0, 1.0, 0.9
        elif k == 4:
            br, bg, bb2, ar2, ag2, ab2 = 0.2, 0.65, 0.45, 1.0, 1.0, 1.0
        else:
            br, bg, bb2, ar2, ag2, ab2 = 0.95, 0.9, 0.85, 0.15, 0.3, 0.7
        # layout: photo block on one side, a real Japanese headline + a small sub line on the other
        m2 = 0.0
        side = 1 if (seed // 7) % 2 == 0 else 0
        pu = uu if side else 1.0 - uu
        if pu < 0.36 and vv > 0.1 and vv < 0.9:
            ph = _hash(seed, int(pu * 7), int(vv * 5), 3)
            m2 = 0.5 + 0.5 * ph
        else:
            hbm = hb3 - 1.4
            ucm = (0.695 if side else 0.305) * pw_       # text block centre (m, along the panel)
            ent = (seed // 3) % (HA.shape[1] // 2)
            um = uu * pw_
            for ln in range(2):
                e2 = 2 * ent + ln
                vt, vb_ = (0.84, 0.5) if ln == 0 else (0.38, 0.2)
                lh = (vt - vb_) * hbm                    # line height (m)
                sc = min(lh / R_TX, 0.5 * pw_ / max(HN[e2], 1.0))   # m per texel (fit height and width)
                vm = vv * hbm
                ty = (vt * hbm - vm) / sc - 0.5 * (lh / sc - R_TX)
                tx = (um - ucm) / sc + 0.5 * HA.shape[3]
                if ty >= 0.0 and ty <= R_TX:
                    g2 = _tex(HA, e2, ty, tx, fu / sc)
                    m2 = max(m2, g2 * (1.0 if ln == 0 else 0.9))
        cr, cg, cb = br * (1 - m2) + ar2 * m2, bg * (1 - m2) + ag2 * m2, bb2 * (1 - m2) + ab2 * m2
        lit = B[LIT0]
        out[0], out[1], out[2] = cr * 0.35, cg * 0.3, cb * 0.35
        if lit > 0.5:
            li = 0.8 * (0.9 + 0.2 * vv)
            out[3], out[4], out[5] = cr * li, cg * li, cb * li
            out[6] = -10.0
            out[7] = 1.0
        return
    if mat == M_NAME:
        # rooftop company-name board: a real Japanese name in one line, fitted to the panel. Lit ones are
        # internally lit white/cream panels with coloured letters, or dark panels with glowing channel letters
        pw_ = B[BX1] - B[BX0]
        if face != 0:
            out[0], out[1], out[2] = wr * 0.6, wg * 0.6, wb * 0.7
            return
        fr = max(0.12, fu * 0.7)
        if v < fr or v > hb - fr or u < fr or u > pw_ - fr:
            out[0], out[1], out[2] = 0.1, 0.09, 0.14            # thin dark frame
            return
        k = seed % 6
        if k == 0:      # white panel, red letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.95, 0.93, 0.88, 0.8, 0.1, 0.1, False
        elif k == 1:    # white panel, navy letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.9, 0.92, 0.95, 0.06, 0.16, 0.55, False
        elif k == 2:    # dark panel, warm channel letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.1, 0.1, 0.14, 1.0, 0.82, 0.5, True
        elif k == 3:    # green panel, white letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.08, 0.42, 0.28, 1.0, 1.0, 0.95, False
        elif k == 4:    # dark panel, cyan channel letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.12, 0.12, 0.2, 0.5, 0.95, 1.1, True
        else:           # dark panel, pink-red channel letters
            br, bg, bb2, ar2, ag2, ab2, chan = 0.14, 0.1, 0.16, 1.1, 0.35, 0.45, True
        ent = (seed // 6) % NA.shape[1]
        ih = hb - 2 * fr
        iw = pw_ - 2 * fr
        sc = min(ih * 0.78 / R_TX, iw * 0.9 / max(NN[ent], 1.0))
        ty = (hb - fr - v) / sc - 0.5 * (ih / sc - R_TX)
        tx = (u - 0.5 * pw_) / sc + 0.5 * NA.shape[3]
        g = 0.0
        if ty >= 0.0 and ty <= R_TX:
            g = min(_tex(NA, ent, ty, tx, fu / sc) * 1.1, 1.0)
        cr, cg, cb = br * (1 - g) + ar2 * g, bg * (1 - g) + ag2 * g, bb2 * (1 - g) + ab2 * g
        out[0], out[1], out[2] = cr * 0.3, cg * 0.28, cb * 0.33
        if B[LIT0] > 0.5:
            if chan:
                li = 0.95 * g + 0.03 * (1 - g)
            else:
                li = 0.5
            out[3], out[4], out[5] = cr * li, cg * li, cb * li
            out[6] = -10.0
            out[7] = 1.0
        return
    if mat == M_PLAIN or mat == M_TANK or mat == M_FENCE:
        # rooftop machinery / stair housing / water tank / fence: plain painted forms
        if mat == M_TANK:
            # horizontal band seams on a tank, lighter paint
            band = _pulse(v, fv, 1.2, 0.0, 0.12)
            wr, wg, wb = wr * (1.15 - 0.2 * band), wg * (1.15 - 0.2 * band), wb * (1.15 - 0.2 * band)
        tl = _sstep(max(fv * 2.0, 0.1) + 0.2, 0.0, hb - v)
        out[0], out[1], out[2] = wr + tl * 0.12, wg + tl * 0.08, wb + tl * 0.1
        return
    if mat == M_CURTAIN:
        # reflective glass curtain wall: mirrors the orange -> pink afterglow sky and cloud, with a broad soft
        # vertical specular streak + a thin sharp one, faint mullion grid, very few lights seen through it
        hb2 = max(hb, 1.0)
        q = min(max(v / hb2, 0.0), 1.0)
        wdt = max(B[BX1] - B[BX0], 1.0) if face == 0 else max(B[BZ1] - B[BZ0], 1.0)
        uu = u / wdt
        if face == 2:
            gr, gg, gb = 0.05, 0.06, 0.16
        else:
            # reflected sky: violet at the base, orange-pink band, magenta-violet up top (painted, not physical)
            if q < 0.35:
                gr, gg, gb = _lerp3(0.16, 0.1, 0.26, 0.62, 0.28, 0.36, q / 0.35)
            elif q < 0.7:
                gr, gg, gb = _lerp3(0.62, 0.28, 0.36, 0.95, 0.46, 0.34, (q - 0.35) / 0.35)
            else:
                gr, gg, gb = _lerp3(0.95, 0.46, 0.34, 0.56, 0.3, 0.55, (q - 0.7) / 0.3)
            if face == 1:
                gr, gg, gb = gr * 1.25 + 0.1, gg * 1.15 + 0.04, gb * 1.0
            # reflected cloud shapes: a couple of soft diagonal bright bands
            cl = _glass_refl(u, v + B[BASE], seed, math.floor(v / B[FLH]), math.floor(u / B[BAYW]))
            if face == 1:
                gr, gg, gb = gr * (1 - 0.5 * cl) + 1.1 * cl * 0.5, gg * (1 - 0.5 * cl) + 0.62 * cl * 0.5, gb * (1 - 0.5 * cl) + 0.5 * cl * 0.5
            else:
                gr, gg, gb = gr * (1 - 0.45 * cl) + 0.62 * cl * 0.45, gg * (1 - 0.45 * cl) + 0.42 * cl * 0.45, gb * (1 - 0.45 * cl) + 0.66 * cl * 0.45
            # vertical specular streaks
            sp0 = 0.12 + 0.25 * _hash(seed, 4, 5, 6)
            sw_ = 0.07 + 0.05 * _hash(seed, 7, 8, 9)
            st = math.exp(-((uu - sp0) / sw_) ** 2) * (0.35 + 0.65 * q)
            st2 = math.exp(-((uu - sp0 - 0.16) / 0.012) ** 2) * (0.3 + 0.7 * q)
            gr += 0.9 * st + 0.9 * st2
            gg += 0.55 * st + 0.62 * st2
            gb += 0.35 * st + 0.45 * st2
        # mullions (vertical every bay) and floor lines, faint
        flh_ = B[FLH]
        bay_ = B[BAYW]
        ml = _pulse(u, fu, bay_, 0.0, 0.25)
        fl_ = _pulse(v, fv, flh_, 0.0, 0.55)
        dk = 1.0 - 0.38 * max(ml, fl_)
        if face == 0:
            # GLASS < 1 dims the camera-facing wall (round 7: the podium mirrors the dimmer eastern sky)
            gk = B[GLASS]
            gr, gg, gb = gr * gk, gg * gk, gb * (0.35 + 0.65 * gk)
        out[0], out[1], out[2] = gr * dk, gg * dk, gb * dk
        # a few lit offices seen through the tinted glass (switch on progressively)
        flo = math.floor(v / flh_)
        ci = math.floor(u / bay_)
        if v > flh_ and v < hb - 2.0:
            sec = int(ci // 5)
            r1 = _hash(seed, int(flo) // 2, sec, 77)
            on = 1e9
            if r1 < B[LIT0] * 0.6:
                on = -10.0
            elif r1 < (B[LIT0] + B[LITADD]) * 0.6:
                on = 0.3 + 4.6 * _hash(seed, int(flo) // 2, sec, 78)
            if on < 1e8:
                wm = _pulse(v - 0.4, fv, flh_, 0.0, flh_ * 0.7) * (1.0 - ml)
                out[3], out[4], out[5] = 0.75 * wm, 0.62 * wm, 0.42 * wm
                out[6] = on
                out[7] = wm
        return
    if mat == M_DARK:
        # dark granite / smoked-glass tower: near-black facade with small punched windows, lit individually
        # and sparsely (scattered warm offices that switch on one by one)
        flh_ = B[FLH]
        bay_ = B[BAYW]
        fl_ = math.floor(v / flh_)
        ci = math.floor(u / bay_)
        wu = _pulse(u - bay_ * 0.25, fu, bay_, 0.0, bay_ * 0.5)
        wv = _pulse(v - flh_ * 0.3, fv, flh_, 0.0, flh_ * 0.45)
        if v < flh_ or v > hb - 2.5:
            wv = 0.0
        wm = wu * wv
        if face == 1:
            mr, mg, mb_ = _sky_mirror(v + B[BASE])
            cl = 0.5 + 0.3 * math.sin(u * 0.09 + seed * 0.13) + 0.3 * math.sin((v + B[BASE]) * 0.035 - u * 0.05 + seed * 0.71)
            cl = _sstep(0.55, 0.85, cl)
            br, bg_, bb_ = 0.14 + 0.3 * mr + 0.2 * cl, 0.08 + 0.2 * mg + 0.12 * cl, 0.14 + 0.2 * mb_ + 0.08 * cl
        elif face == 0:
            q_ = min(max(v / max(hb, 1.0), 0.0), 1.0)
            cl = _glass_refl(u, v + B[BASE], seed, fl_, ci)
            br, bg_, bb_ = 0.045 + 0.12 * q_ * q_ + 0.2 * cl, 0.05 + 0.06 * q_ * q_ + 0.13 * cl, 0.11 + 0.12 * q_ * q_ + 0.2 * cl
        else:
            br, bg_, bb_ = 0.03, 0.035, 0.09
        # glass of the windows mirrors a little sky
        gr, gg, gb = (br * 1.5, bg_ * 1.5, bb_ * 1.4) if face == 1 else (0.1, 0.1, 0.2)
        # floor slabs + mullions: dark lines, the slab top catching a warm line on the lit face
        sl_ = _pulse(v, fv, flh_, 0.0, 0.5)
        ml_ = _pulse(u, fu, bay_, 0.0, 0.25)
        dk_ = 1.0 - 0.35 * max(sl_, ml_)
        out[0] = (br * (1 - wm) + gr * wm) * dk_
        out[1] = (bg_ * (1 - wm) + gg * wm) * dk_
        out[2] = (bb_ * (1 - wm) + gb * wm) * dk_
        if face == 1:
            st_ = _pulse(v, fv, flh_, 0.3, 0.22)
            out[0] += 0.22 * st_
            out[1] += 0.11 * st_
            out[2] += 0.06 * st_
        # offices light up in blocks (3 floors x 7 bays), not as random dots; a few dark windows inside
        cf_ = int(fl_) // 3
        cc_ = int(ci) // 7
        r1 = _hash(seed, cf_, cc_, 55 + face)
        on = 1e9
        if r1 < B[LIT0] * 0.9:
            on = -10.0
        elif r1 < (B[LIT0] + B[LITADD]) * 0.9:
            on = 0.2 + 5.0 * _hash(seed, cf_, cc_, 56) + 0.1 * (int(fl_) % 3)
        if on < 1e8 and _hash(seed, int(fl_), int(ci), 58) < 0.12:
            on = 1e9
        if on < 1e8:
            r3 = _hash(seed, int(fl_), int(ci), 57)
            if r3 < 0.7:
                er, eg, eb = 1.2, 0.68, 0.3
            else:
                er, eg, eb = 1.05, 0.98, 0.85
            out[3], out[4], out[5] = er * wm, eg * wm, eb * wm
            out[6] = on
            out[7] = wm
        return
    flh = B[FLH]
    bay = B[BAYW]
    wwf = B[WWF]
    whf = B[WHF]
    glassy = B[GLASS]
    # ---- glass reflection colours
    if face == 0:
        if mat == M_GLASS or mat == M_DARK:
            # east-facing glass mirrors the eastern dusk sky: dark blue earth shadow low, the pink belt of
            # Venus in the middle, violet above (mapped along the building's own height)
            q = min(max(v / max(hb, 1.0), 0.0), 1.0)
            if q < 0.5:
                gr, gg, gb = _lerp3(0.04, 0.07, 0.16, 0.1, 0.13, 0.28, q / 0.5)
            elif q < 0.72:
                gr, gg, gb = _lerp3(0.1, 0.13, 0.28, 0.36, 0.22, 0.38, (q - 0.5) / 0.22)
            else:
                gr, gg, gb = _lerp3(0.36, 0.22, 0.38, 0.16, 0.17, 0.38, (q - 0.72) / 0.28)
        else:
            k = _sstep(40.0, 320.0, vb)
            gr, gg, gb = 0.05 + 0.12 * k, 0.08 + 0.12 * k, 0.16 + 0.2 * k
    elif face == 1:
        gr, gg, gb = _sky_mirror(vb)
        # the pink-gold cumulus / cloud streaks reflected as big soft shapes across the west glass
        cl = 0.5 + 0.3 * math.sin(u * 0.09 + seed * 0.13) + 0.3 * math.sin(vb * 0.035 - u * 0.05 + seed * 0.71)
        cl = _sstep(0.55, 0.85, cl)
        gr, gg, gb = gr * (0.8 + 0.1 * cl) + 0.3 * cl, gg * (0.8 + 0.1 * cl) + 0.2 * cl, gb * 0.85 + 0.1 * cl
    else:
        gr, gg, gb = 0.04, 0.06, 0.15
    if (mat == M_GLASS or mat == M_FINS or mat == M_BAND) and face != 2 and hb > 30.0:
        cl = _glass_refl(u, vb, seed, math.floor(v / B[FLH]), math.floor(u / B[BAYW]))
        if face == 1:
            gr, gg, gb = gr * (1 - 0.5 * cl) + 0.6 * cl, gg * (1 - 0.5 * cl) + 0.34 * cl, gb * (1 - 0.5 * cl) + 0.26 * cl
        else:
            gr, gg, gb = gr * (1 - 0.45 * cl) + 0.3 * cl, gg * (1 - 0.45 * cl) + 0.2 * cl, gb * (1 - 0.45 * cl) + 0.32 * cl
    if (mat == M_GLASS or mat == M_DARK) and face == 0:
        # broad diagonal sheen of the pink clouds reflected across the curtain wall
        sh = 0.5 + 0.5 * math.sin(u * 0.05 + vb * 0.022 + seed * 0.37)
        sh = sh ** 4
        gr += 0.16 * sh
        gg += 0.06 * sh
        gb += 0.1 * sh
    # ---- pattern
    nfl = max(math.floor(hb / flh), 1.0)
    top = nfl * flh
    fl = math.floor(v / flh)
    col_i = math.floor(u / bay)
    if mat == M_FINS:
        # vertical stone fins between tall glass slots: strong vertical stripes
        inwin_u = _pulse(u - bay * (1 - wwf) * 0.5, fu, bay, 0.0, bay * wwf)
        inwin_v = _pulse(v - 0.5 * flh * (1 - whf), fv, flh, 0.0, flh * whf)
    elif mat == M_BAND:
        inwin_u = _pulse(u - bay * 0.02, fu, bay, 0.0, bay * 0.96)
        inwin_v = _pulse(v - 0.5 * flh * (1 - whf), fv, flh, 0.0, flh * whf)
    else:
        inwin_v = _pulse(v - 0.5 * flh * (1 - whf), fv, flh, 0.0, flh * whf)
        inwin_u = _pulse(u - bay * (1 - wwf) * 0.5, fu, bay, 0.0, bay * wwf)
    if v < flh * 0.9 or v > top - 0.3:
        inwin_v = 0.0
    wm = inwin_u * inwin_v
    # ---- wall material
    if mat == M_APT:  # apartment: bright balcony parapet bands + dividing walls
        band = _pulse(v, fv, flh, 0.0, flh * 0.3)
        wr, wg, wb = wr * (1 + 0.45 * band), wg * (1 + 0.45 * band), wb * (1 + 0.4 * band)
        wm = wm * (1 - band)
    elif mat == M_CONC:
        band = _pulse(v, fv, flh, 0.0, 0.35)
        wr, wg, wb = wr * (1 + 0.18 * band), wg * (1 + 0.18 * band), wb * (1 + 0.15 * band)
    elif mat == M_FINS:
        # fins catch light on their sunward edge
        wr, wg, wb = wr * 1.15, wg * 1.15, wb * 1.12
    elif mat == M_BAND:
        # light tile spandrels between the ribbon windows
        wr, wg, wb = wr * 1.35, wg * 1.3, wb * 1.25
    if (mat == M_GLASS or mat == M_DARK) and (int(fl) % 12) == 11:
        wm = 0.0
        wr, wg, wb = wr * 0.7, wg * 0.7, wb * 0.75
    if mat == M_GLASS or mat == M_DARK:
        gm = 0.5 if mat == M_GLASS else 0.75
        wr = wr * (1 - gm) + gr * gm * glassy
        wg = wg * (1 - gm) + gg * gm * glassy
        wb = wb * (1 - gm) + gb * gm * glassy
        # faint vertical mullion lines
        ml = _pulse(u, fu, bay, 0.0, 0.12)
        wr, wg, wb = wr * (1 - 0.25 * ml), wg * (1 - 0.25 * ml), wb * (1 - 0.2 * ml)
    # ---- lights: whole office floors / tenant sections switch together (runs of floors), clusters of
    #      neighbouring runs switch on in a wave climbing the building
    grp = max(int(B[GRP]), 1)
    runf = max(int(B[RUNF]), 1)
    sec = int(math.floor(col_i / grp))
    run = int(math.floor(fl / runf))
    tall = hb > 60.0
    if tall:
        # towers: whole tenant floors / half floors light together, in runs of 2-4 floors
        sec = int(math.floor(col_i / (grp * 2 + 6)))
        run = int(math.floor(fl / (runf + 2)))
    r1 = _hash(seed, face + 7, run, sec)
    lit0 = B[LIT0]
    litadd = B[LITADD]
    on = 1e9
    if r1 < lit0:
        on = -10.0
    elif r1 < lit0 + litadd:
        clu = run // 4
        base_t = 0.25 + 4.6 * _hash(seed, clu, sec, 41)
        on = base_t + 0.09 * (run - clu * 4) + 0.05 * _hash(seed, run, sec, 43)
    # light colour per section: fluorescent warm-white / tungsten orange / a few cool blue LED
    lc = B[LITCOL]
    r3 = _hash(seed, run, sec, 99)
    r2 = _hash(seed, int(fl) + 31, int(col_i), 5 + face)
    if r3 < lc:
        er, eg, eb = 1.15, 0.62 + 0.1 * r2, 0.26 + 0.1 * r2
    elif r3 < lc + (1 - lc) * 0.8:
        er, eg, eb = 1.05, 0.98, 0.86
    else:
        er, eg, eb = 0.62, 0.82, 1.15
    ei = 1.0 + 0.35 * r2
    rb_ = _hash(seed, int(fl), int(col_i), 3)
    if rb_ < 0.18:
        ei *= 0.5       # blinds
    elif rb_ < 0.24:
        ei *= 1.45      # a desk lamp / screen right at the glass
    # round 5: varied lighting inside a lit section - empty offices, dimmer back rooms, tinted panes
    rv_ = _hash(seed, int(fl) + 11, int(col_i), 71 + face)
    if rv_ < (0.05 if tall else 0.16):
        on = 1e9
    elif rv_ < (0.15 if tall else 0.34):
        ei *= 0.55
    rt_ = _hash(seed, int(fl), int(col_i) + 5, 73)
    if rt_ < 0.12:
        er, eg, eb = er * 0.85, eg * 0.95, min(eb * 1.35, 1.2)
    elif rt_ > 0.9:
        er, eg, eb = er * 1.05, eg * 0.8, eb * 0.6
    vr = (0.85 + 0.3 * r3) * (0.9 + 0.2 * _hash(seed, int(fl), int(col_i), 91))
    gsr, gsg, gsb = gr * vr, gg * vr, gb * vr
    # round 7: break the regular window grid - whole floors differ (empty dark floors mirror the sky,
    # fully-lit open-plan office floors read as bright ribbons), runs of windows have blinds drawn
    # (pale panels, lit ones glow softer), and a few tenants have warm vs cool interiors
    if mat != M_APT and mat != M_CONC and hb > 30.0:
        fs_ = _hash(seed, int(fl), face, 203)
        if fs_ < 0.13:
            on = 1e9
            gsr, gsg, gsb = gr * 0.75, gg * 0.75, gb * 0.8
        elif fs_ > 0.9 and v > flh and v < top - flh:
            on = -10.0
            er, eg, eb = 1.05, 1.0, 0.9
            ei = 0.95 + 0.1 * r2
    bl_ = _hash(seed, int(fl), int(col_i) // 4, 207)
    if bl_ < 0.24:
        # blinds: a pale fabric panel over the upper part (or all) of the pane
        bf_ = 0.35 + 0.65 * _hash(seed, int(fl), int(col_i) // 4, 208)
        wv0 = 0.5 * flh * (1 - whf)
        rel = (v - fl * flh - wv0) / max(flh * whf, 0.1)
        if rel > 1.0 - bf_:
            if face == 1:
                pr_, pg_, pb_ = 0.62, 0.42, 0.4
            elif face == 0:
                pr_, pg_, pb_ = 0.2, 0.2, 0.3
            else:
                pr_, pg_, pb_ = 0.08, 0.08, 0.16
            gsr, gsg, gsb = gsr * 0.3 + pr_ * 0.7, gsg * 0.3 + pg_ * 0.7, gsb * 0.3 + pb_ * 0.7
            ei *= 0.6
            er, eg, eb = er * 1.02, eg * 0.92, eb * 0.78
    if mat == M_APT and face != 1 and _hash(seed, int(fl), int(col_i), 17) < 0.25:
        gsr, gsg, gsb = lr * 0.95, lg * 0.85, lb * 0.8
    out[0] = wr * (1 - wm) + gsr * wm
    out[1] = wg * (1 - wm) + gsg * wm
    out[2] = wb * (1 - wm) + gsb * wm
    if hb > 45.0 and mat != M_APT and mat != M_CONC:
        # crisp floor-band (slab edge) + mullion lines on tower glass; the slab tops on the west face catch
        # a thin warm line of the afterglow
        slb = _pulse(v, fv, flh, 0.0, 0.5)
        mul = _pulse(u, fu, bay, 0.0, 0.25)
        dk2 = 1.0 - 0.35 * max(slb, mul)
        out[0] *= dk2
        out[1] *= dk2
        out[2] *= dk2
        if face == 1:
            stp = _pulse(v, fv, flh, 0.28, 0.2)
            out[0] += 0.2 * stp
            out[1] += 0.1 * stp
            out[2] += 0.05 * stp
    out[3] = er * ei * wm
    out[4] = eg * ei * wm
    out[5] = eb * ei * wm
    out[6] = on
    out[7] = wm
    if on > 1e8:
        out[3] = 0.0
        out[4] = 0.0
        out[5] = 0.0
        out[7] = 0.0
    # ---- lit crown band (a few towers): a ring of light under the parapet
    if B[CROWN] > 0.5 and v > hb - 3.2 and v < hb - 1.2:
        cu = _pulse(u, fu, 1.4, 0.0, 0.9)
        out[3], out[4], out[5] = 1.2 * cu + out[3] * (1 - cu), 1.05 * cu + out[4] * (1 - cu), 0.85 * cu + out[5] * (1 - cu)
        out[6] = -10.0
        out[7] = max(out[7], cu)


@njit(cache=True)
def _sign(u, v, fu, B, x0, x1, hb, out, VA, VN):
    """Vertical shop / building sign strip on the front face with a real Japanese word (tategaki) sampled
    from the pre-rendered font atlas VA. Returns True if (u, v) is on the sign (out filled)."""
    sgn = int(B[SIGN])
    wdt = x1 - x0
    sw = min(max(0.16 * wdt, 2.0), 3.6)
    us = 0.05 * wdt if (sgn // 5) % 2 == 0 else wdt - 0.05 * wdt - sw
    v0 = 3.5
    cell = sw * 0.95                  # character pitch (m)
    pad = 0.3
    nfit = int((hb * 0.85 - v0 - 2 * pad) / cell)
    nv = VA.shape[1]
    # prefer a word of a target length (tall signs on tall buildings), else the longest that fits
    nt = min(nfit, 3 + sgn % 6)
    idx = -1
    best = 0
    for k in range(nv):
        j = (sgn + k * 7) % nv
        if VN[j] == nt:
            idx = j
            break
        if VN[j] <= nfit and VN[j] > best:
            best = VN[j]
            idx = j
    if idx < 0:
        return False
    n = VN[idx]
    v1 = v0 + 2 * pad + n * cell
    if u < us or u > us + sw or v < v0 or v > v1:
        return False
    gu = (u - us) / sw
    ks = sgn % 5
    # style: bg colour, glyph colour, lit bg?
    if ks == 0:
        br, bg_, bb = 1.0, 0.95, 0.85
        cr, cg, cb = 0.8, 0.08, 0.08
        litbg = True
    elif ks == 1:
        br, bg_, bb = 0.85, 0.1, 0.12
        cr, cg, cb = 1.1, 1.05, 0.95
        litbg = False
    elif ks == 2:
        br, bg_, bb = 1.0, 0.82, 0.25
        cr, cg, cb = 0.15, 0.1, 0.12
        litbg = True
    elif ks == 3:
        br, bg_, bb = 0.1, 0.25, 0.7
        cr, cg, cb = 1.0, 1.0, 1.05
        litbg = False
    else:
        br, bg_, bb = 0.12, 0.1, 0.14
        cr, cg, cb = 0.4, 1.0, 0.75
        litbg = False
    # frame border
    if gu < 0.06 or gu > 0.94 or v - v0 < 0.12 or v1 - v < 0.12:
        out[0], out[1], out[2] = 0.12, 0.1, 0.16
        out[3], out[4], out[5] = 0.0, 0.0, 0.0
        out[6] = -10.0
        out[7] = 0.0
        return True
    sc = cell / R_TX                  # m per texel
    ty = (v1 - pad - v) / sc
    tx = (u - us - 0.5 * sw) / sc + 0.5 * R_TX
    g = _tex(VA, idx, ty, tx, fu / sc) if ty < n * R_TX else 0.0
    g = min(g * 1.15, 1.0)
    li = 1.0 + 0.2 * _hash(sgn, 5, 5, 5)
    rr = br * (1 - g) + cr * g
    rg = bg_ * (1 - g) + cg * g
    rb = bb * (1 - g) + cb * g
    out[0], out[1], out[2] = rr * 0.3, rg * 0.3, rb * 0.3
    if litbg:
        out[3], out[4], out[5] = rr * li * 1.1, rg * li * 1.1, rb * li * 1.1
    else:
        out[3], out[4], out[5] = cr * g * li * 1.2 + br * (1 - g) * 0.35, cg * g * li * 1.2 + bg_ * (1 - g) * 0.35,             cb * g * li * 1.2 + bb * (1 - g) * 0.35
    out[6] = -10.0
    out[7] = 1.0
    return True


# ---------------------------------------------------------------------------------------------------------------
# Round 10: painted tower facades. Each tower carries a palette / material family (PAL column) instead of one
# shared pink glass: blue / teal / bronze / smoked curtain walls, white aluminium grids, beige stone with punched
# windows, silver fins. The glass mirrors the sky per panel (every mirror panel is tilted a hair, so the
# reflection steps from panel to panel), picks up the reflected hazy skyline as a jagged dark band, and the
# structure is drawn: mullions, spandrels, piers, corner returns, mechanical floors, lobby and parapet.
# Face lighting: the camera-facing (east) face is in shadow under the cool sky dome; the -x face looks at the
# afterglow and mirrors hot gold.

# palette table: frame rgb, glass tint rgb, reflectivity, spandrel frac, mullion w (m), pier every n bays (0 none),
# pier w (m), corner return w (bays), punched (0 curtain / 1 punched stone)
PALS = np.array([
    # frame               tint                refl  spf   mul   np   pw    cw   punched
    [0.30, 0.36, 0.48, 0.55, 0.72, 1.00, 0.92, 0.16, 0.10, 0.0, 0.0, 0.0, 0.0],   # 0 blue curtain wall
    [0.86, 0.87, 0.92, 0.52, 0.62, 0.82, 0.70, 0.40, 0.30, 0.0, 0.0, 1.0, 0.0],   # 1 white aluminium grid
    [0.26, 0.38, 0.40, 0.42, 0.86, 0.84, 0.90, 0.20, 0.14, 4.0, 0.5, 0.0, 0.0],   # 2 teal-green glass
    [0.82, 0.70, 0.58, 0.58, 0.58, 0.66, 0.55, 0.48, 0.55, 3.0, 0.7, 1.0, 1.0],   # 3 beige stone, punched
    [0.13, 0.13, 0.17, 0.44, 0.48, 0.62, 0.80, 0.26, 0.20, 0.0, 0.0, 0.5, 0.0],   # 4 dark granite / smoked
    [0.34, 0.26, 0.22, 0.92, 0.70, 0.52, 0.88, 0.22, 0.14, 6.0, 0.4, 0.0, 0.0],   # 5 bronze glass
    [0.64, 0.66, 0.72, 0.52, 0.64, 0.86, 0.80, 0.12, 0.42, 1.0, 0.42, 1.0, 0.0],  # 6 silver vertical fins
    [0.70, 0.62, 0.66, 0.56, 0.60, 0.78, 0.62, 0.52, 0.45, 2.0, 0.5, 1.0, 1.0],   # 7 pale rose tile, punched
])
NPAL = 8


@njit(cache=True, inline='always')
def _skyrefl(s, face, br):
    """reflected sky / hazy horizon colour at reflected elevation s (0 = horizon, + up, - down)"""
    if face == 1:
        # the afterglow side: hot gold at the horizon, orange then rose-violet higher up
        if s >= 0.0:
            if s < 0.6:
                r, g, b = _lerp3(1.45, 0.86, 0.40, 1.15, 0.52, 0.34, s / 0.6)
            elif s < 2.0:
                r, g, b = _lerp3(1.15, 0.52, 0.34, 0.62, 0.36, 0.52, (s - 0.6) / 1.4)
            else:
                r, g, b = 0.62, 0.36, 0.52
        else:
            # below the horizon: the backlit city through the haze, glowing near the horizon
            k = math.exp(s * 1.6)
            r, g, b = _lerp3(0.16, 0.10, 0.16, 1.0, 0.58, 0.36, k)
    else:
        # the eastern dusk sky behind the camera: pink-lavender belt of Venus low, cool blue above
        if s >= 0.0:
            if s < 0.7:
                r, g, b = _lerp3(0.58, 0.50, 0.66, 0.40, 0.46, 0.72, s / 0.7)
            elif s < 2.4:
                r, g, b = _lerp3(0.40, 0.46, 0.72, 0.20, 0.27, 0.56, (s - 0.7) / 1.7)
            else:
                r, g, b = 0.20, 0.27, 0.56
        else:
            k = math.exp(s * 1.4)
            r, g, b = _lerp3(0.07, 0.08, 0.15, 0.40, 0.38, 0.55, k)
    return r * br, g * br, b * br


@njit(cache=True)
def _tower(u, v, fu, fv, B, face, z, cam_h, sunx, out):
    """Painted high-rise facade (see above). u, v face coords (m), fu, fv pixel footprint (m).
    Writes out[0:3] colour, out[3:6] emission, out[6] on-time, out[7] emission weight."""
    seed = int(B[SEED])
    pal = ((int(B[PAL]) - 1) % 16) % NPAL
    P = PALS[pal]
    hb = B[BH] - B[BASE]
    vb = v + B[BASE]
    wdt = (B[BX1] - B[BX0]) if face != 1 and face != 2 else (B[BZ1] - B[BZ0])
    wdt = max(wdt, 1.0)
    flh = B[FLH]
    bay = B[BAYW]
    if pal == 3 or pal == 7:
        bay = max(bay, 2.6)
    tseed = int(B[PAL]) // 16                        # tower id: shared by all tiers of one tower
    # ---- face lighting (painted): shadow face cool, afterglow face warm, far face deep blue
    hv = min(max(vb / 260.0, 0.0), 1.0)
    if face == 0:
        lr, lg, lb = 0.30 + 0.10 * hv, 0.33 + 0.10 * hv, 0.50 + 0.12 * hv
    elif face == 1:
        lr, lg, lb = 0.95 + 0.35 * hv, 0.58 + 0.16 * hv, 0.42 + 0.06 * hv
    else:
        lr, lg, lb = 0.10, 0.12, 0.24
    ab = (B[AR] + B[AG] + B[AB]) / 3.0
    vk = 0.85 + 0.3 * (ab - 0.85)                          # per-tower value
    fr_, fg_, fb_ = P[0] * lr * vk, P[1] * lg * vk, P[2] * lb * vk
    # ---- reflected elevation, per mirror panel tilt
    pf = 2 + (tseed % 2)                                    # floors per panel
    pb = 1 + (tseed // 3) % 3                               # bays per panel
    pfl = math.floor(v / (flh * pf))
    pcl = math.floor(u / (bay * pb))
    jit = (_hash(seed, int(pfl), int(pcl), 401) - 0.5)
    jamp = 0.55 * _sstep(bay * pb * 1.2, bay * pb * 0.4, fu)  # fades when a panel shrinks below a pixel
    s = (cam_h - vb) / max(z, 1.0) * 40.0
    s += 0.35 * math.sin(u * 0.045 + tseed * 0.7) + 0.2 * math.sin(u * 0.13 - v * 0.02 + tseed)
    s += jit * jamp
    # reflected skyline of the city behind the camera: a jagged darker band just above the horizon
    sk = 0.25 + 0.3 * _hash(tseed, int(math.floor(u / 9.0)), 0, 403) + 0.25 * math.sin(u * 0.021 + tseed)
    br = 0.92 + 0.16 * _hash(seed, int(pfl), int(pcl), 402) * (jamp / 0.55)
    gr, gg, gb = _skyrefl(s, face, br)
    if s < sk and s > -1.5:
        m = _sstep(sk, sk - 0.08, s)
        if face == 1:
            cr_, cg_, cb_ = 0.34, 0.18, 0.2
        else:
            cr_, cg_, cb_ = 0.12, 0.12, 0.22
        gr, gg, gb = gr * (1 - 0.45 * m) + cr_ * 0.45 * m, gg * (1 - 0.45 * m) + cg_ * 0.45 * m, gb * (1 - 0.45 * m) + cb_ * 0.45 * m
    # reflected cloud masses (soft, large) on the shadow face
    if face == 0:
        c = 0.5 + 0.3 * math.sin(u * 0.03 + vb * 0.012 + tseed) + 0.25 * math.sin(u * 0.011 - vb * 0.02 + tseed * 1.3)
        cm = _sstep(0.62, 0.72, c) * _sstep(-0.2, 0.8, s)
        gr, gg, gb = gr + 0.16 * cm, gg + 0.1 * cm, gb + 0.12 * cm
    refl = P[6]
    tr, tg, tb = P[3], P[4], P[5]
    # glass = mirrored sky tinted by the glass + a little dark interior
    if face == 2:
        gr, gg, gb = 0.06, 0.07, 0.15
    xr_ = gr * tr * refl + 0.03 * (1 - refl)
    xg_ = gg * tg * refl + 0.035 * (1 - refl)
    xb_ = gb * tb * refl + 0.07 * (1 - refl)
    # ---- structure masks
    spf = P[7]
    mw = P[8]
    punched = P[12] > 0.5
    fl = math.floor(v / flh)
    ci = math.floor(u / bay)
    pu = _pulse(u - 0.5 * mw, fu, bay, 0.0, bay - mw)
    pv = _pulse(v - spf * flh * 0.6, fv, flh, 0.0, flh * (1.0 - spf))
    gm = pu * pv
    # piers every n bays (a deeper vertical member, lighter on its lit edge)
    npb = int(P[9])
    pier = 0.0
    if npb > 0:
        pier = _pulse(u + 0.5 * P[10], fu, bay * npb, 0.0, P[10])
    # corner returns: solid frame at both face edges on framed / stone towers
    cw = P[11] * bay
    corner = 0.0
    if cw > 0.0:
        corner = max(_sstep(cw + fu, cw, u), _sstep(wdt - cw - fu, wdt - cw, u))
    # mechanical floors every mf floors: louvred band, no windows
    mf = 10 + tseed % 7
    mech = 1.0 if (int(fl) % mf) == mf - 1 and v > flh * 3 and v < hb - flh * 2 else 0.0
    # parapet + lobby
    par = _sstep(hb - 1.8 - fv, hb - 1.8, v)
    lob = 1.0 - _sstep(0.0, 0.5, vb - 9.0)
    solid = max(max(pier, corner), max(par, mech))
    gm = gm * (1.0 - solid)
    # vertical recessed notch in the middle of some faces (reads as depth)
    notch = 0.0
    if (tseed // 5) % 3 == 0 and wdt > 20.0:
        nw = bay * (1 + (tseed // 7) % 2)
        notch = _pulse(u - 0.5 * wdt + 0.5 * nw, fu, 1e6, 0.0, nw)
    # ---- spandrel colour: opaque glass on curtain walls, frame on punched / framed walls
    if punched:
        sr, sg, sb = fr_, fg_, fb_
        cl_ = _pulse(v, fv, flh, 0.0, 0.12)
        sr, sg, sb = sr * (1 - 0.18 * cl_), sg * (1 - 0.18 * cl_), sb * (1 - 0.15 * cl_)
    else:
        sr, sg, sb = xr_ * 0.62 + fr_ * 0.25, xg_ * 0.62 + fg_ * 0.25, xb_ * 0.62 + fb_ * 0.25
    spm = 1.0 - pv                # spandrel rows
    mulm = max(pv - pu * pv, 0.0)  # just the mullions
    cr = xr_ * gm + (sr * spm + fr_ * mulm) * (1 - solid)
    cg = xg_ * gm + (sg * spm + fg_ * mulm) * (1 - solid)
    cb = xb_ * gm + (sb * spm + fb_ * mulm) * (1 - solid)
    sk2 = 1.0
    if face == 1 and pier > 0.0:
        sk2 = 1.25
    cr += fr_ * solid * sk2
    cg += fg_ * solid * sk2
    cb += fb_ * solid * sk2
    if mech > 0.0:
        lv = _pulse(v, fv, 0.45, 0.0, 0.15)
        cr, cg, cb = cr * (0.8 - 0.2 * lv), cg * (0.8 - 0.2 * lv), cb * (0.82 - 0.2 * lv)
    if notch > 0.0:
        dn = 0.55 if face == 0 else 0.7
        cr, cg, cb = cr * (1 - notch * (1 - dn)), cg * (1 - notch * (1 - dn)), cb * (1 - notch * (1 - dn) * 0.8)
    if lob > 0.0:
        cr, cg, cb = cr * (1 - lob) + 0.12 * lob, cg * (1 - lob) + 0.1 * lob, cb * (1 - lob) + 0.14 * lob
    # broad painted value gradient: darker toward the base, a touch of sky light at the top
    vg = 0.78 + 0.3 * _sstep(0.0, 1.0, v / max(hb, 1.0)) if B[BASE] < 1.0 else 1.05
    out[0], out[1], out[2] = cr * vg, cg * vg, cb * vg
    if face == 0:
        # light wraps round from the afterglow: the sunward third of the shadow face warms and lifts
        sw = 1.0 if B[BX0] + B[BX1] > 2.0 * sunx * z else 0.0
        du = u / wdt if sw > 0.5 else 1.0 - u / wdt
        wr = (1.0 - _sstep(0.0, 0.45, du)) * (0.35 + 0.65 * hv)
        out[0] += wr * 0.16 * (0.4 + out[0])
        out[1] += wr * 0.07 * (0.4 + out[1])
        out[2] += wr * 0.02
    # ---- lights: tenants switch on whole floor sections; warm white / fluorescent / a few cool
    out[3], out[4], out[5], out[6], out[7] = 0.0, 0.0, 0.0, 1e9, 0.0
    if face == 2 or gm <= 0.0:
        return
    grp = 2 + (seed % 5)
    sec = int(math.floor((ci + 3.0 * _hash(seed, int(fl), 0, 409)) / grp))   # sections staggered per floor
    runf = 1
    run = int(math.floor(fl / runf))
    r1 = _hash(seed, face + 7, run, sec)
    lit0 = B[LIT0]
    litadd = B[LITADD]
    on = 1e9
    if r1 < lit0:
        on = -10.0
    elif r1 < lit0 + litadd:
        clu = run // 4
        on = 0.25 + 4.6 * _hash(seed, clu, sec, 41) + 0.09 * (run - clu * 4)
    if _hash(seed, int(fl), int(ci), 405) < 0.3:
        on = 1e9
    if on > 1e8:
        return
    r3 = _hash(seed, run, sec, 99)
    if r3 < 0.45:
        er, eg, eb = 1.05, 0.92, 0.72
    elif r3 < 0.85:
        er, eg, eb = 0.95, 0.97, 0.92
    elif r3 < 0.93:
        er, eg, eb = 1.15, 0.66, 0.3
    else:
        er, eg, eb = 0.62, 0.8, 1.1
    ei = (0.32 + 0.3 * _hash(seed, int(fl), int(ci), 406)) * (1.0 - 0.6 * refl * (1.0 if face == 1 else 0.3))
    # lit panes show the ceiling light as the upper part of the pane (a horizontal strip, not a full block)
    wv0 = spf * flh * 0.6
    rel = (v - fl * flh - wv0) / max(flh * (1.0 - spf), 0.1)
    gm = gm * (0.45 + 0.55 * _sstep(0.25, 0.55, rel))
    if _hash(seed, int(fl), int(ci) // 3, 407) < 0.2:
        ei *= 0.5
    out[3], out[4], out[5] = er * ei * gm, eg * ei * gm, eb * ei * gm
    out[6] = on
    out[7] = gm
    # lit crown ring under the parapet
    if B[CROWN] > 0.5 and B[CROWN] < 1.5 and v > hb - 3.2 and v < hb - 1.2:
        cu = _pulse(u, fu, 1.4, 0.0, 0.9)
        out[3], out[4], out[5] = 1.2 * cu, 1.05 * cu, 0.85 * cu
        out[6] = -10.0
        out[7] = cu


@njit(cache=True, parallel=True)
def render_band(B, order, bb, Hs, Ws, ss, x_off, y_off, cx, hy, f, cam_h, zmin, zmax,
                haze, haze_hy, fogZ, lightZ, ground_on, street, tr_a, tr_b, tr_side, sun_xw, sod, VA, VN, HA, HN, NA, NN):
    """Rasterize buildings B[order] (far->near) into supersampled buffers.
    street: (px, swx, ox, pz, swz, oz) street grid (m). sod: sodium haze colour (3,).
    Returns rgb (Hs, Ws, 3), a (Hs, Ws), E (Hs, Ws, 3), Eon (Hs, Ws)."""
    rgb = np.zeros((Hs, Ws, 3), np.float32)
    al = np.zeros((Hs, Ws), np.float32)
    E = np.zeros((Hs, Ws, 3), np.float32)
    Eon = np.zeros((Hs, Ws), np.float32)
    iz = np.zeros((Hs, Ws), np.float32)     # round 10: inverse depth (for depth-tested spires)
    Hh, Wh = haze.shape[0], haze.shape[1]
    n = order.shape[0]
    PX, SWX, OX, PZ, SWZ, OZ = street[0], street[1], street[2], street[3], street[4], street[5]
    T = 48
    nt = Ws // T + 1
    for py in prange(Hs):
        tmp = np.zeros(8)
        tmp2 = np.zeros(8)
        sy = y_off + (py + 0.5) / ss
        dy = sy - hy
        # ---- per-row candidate lists binned into x tiles (keeps far->near order)
        cand = np.empty(n, np.int64)
        nc = 0
        for oi in range(n):
            i = order[oi]
            if sy >= bb[i, 1] and sy <= bb[i, 3]:
                cand[nc] = i
                nc += 1
        cnt = np.zeros(nt + 1, np.int64)
        for k in range(nc):
            i = cand[k]
            t0 = max(int(math.floor((bb[i, 0] - x_off) * ss / T)), 0)
            t1 = min(int(math.floor((bb[i, 2] - x_off) * ss / T)), nt - 1)
            for tt in range(t0, t1 + 1):
                cnt[tt + 1] += 1
        for tt in range(nt):
            cnt[tt + 1] += cnt[tt]
        lst = np.empty(max(cnt[nt], 1), np.int64)
        fillp = cnt[:nt].copy()
        for k in range(nc):
            i = cand[k]
            t0 = max(int(math.floor((bb[i, 0] - x_off) * ss / T)), 0)
            t1 = min(int(math.floor((bb[i, 2] - x_off) * ss / T)), nt - 1)
            for tt in range(t0, t1 + 1):
                lst[fillp[tt]] = i
                fillp[tt] += 1
        for px in range(Ws):
            sx = x_off + (px + 0.5) / ss
            tile = px // T
            dx = sx - cx
            hit = False
            cr, cg, cb, er, eg, eb, eon, ew = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            zh = 0.0
            yw_hit = 0.0
            # ---- ground inside this band: streets glowing with sodium light
            if ground_on and dy > 0.0:
                zg = f * cam_h / dy
                xg = dx * zg / f
                zt = tr_a + tr_b * xg
                okside = True
                if tr_side == 1 and zg <= zt:
                    okside = False
                if tr_side == 2 and zg > zt:
                    okside = False
                if zg >= zmin and zg < zmax and okside:
                    fp = zg / f / ss
                    fz = fp * zg / max(cam_h, 1.0)
                    sz = _pulse(zg - OZ, fz, PZ, 0.0, SWZ)
                    rowz = math.floor((zg - OZ) / PZ)
                    frz = (rowz * 0.618034 + 0.3) % 1.0
                    oxr = OX + PX * (0.55 * frz - 0.275)
                    sxg = _pulse(xg - oxr, fp, PX, 0.0, SWX)
                    st = max(sz, sxg)
                    cr, cg, cb = 0.05 + 0.12 * st, 0.05 + 0.06 * st, 0.1 + 0.03 * st
                    lamp = _pulse(xg + 3.0 * math.floor(zg / PZ), fp * 2, 24.0, 0.0, 2.2) * sz
                    lamp2 = _pulse(zg, fz, 26.0, 0.0, 2.0) * sxg
                    li = min(lamp + lamp2, 1.0) * 0.45 + st * 0.05
                    er, eg, eb = 1.0 * li, 0.6 * li, 0.28 * li
                    if street.shape[0] > 7 and xg >= street[6] and xg <= street[7]:
                        # round 6: the boulevard - dark asphalt, pale sidewalks under sodium lamps, and
                        # long-exposure light trails: white-gold headlights in the inbound lanes, red tail
                        # lights outbound, bunched by traffic (static: no flicker)
                        ua = xg - street[6]
                        wa = street[7] - street[6]
                        cr, cg, cb = 0.07, 0.065, 0.12
                        er, eg, eb = 0.0, 0.0, 0.0
                        if ua < 3.5 or ua > wa - 3.5:
                            cr, cg, cb = 0.2, 0.15, 0.2
                            lp = _pulse(zg, fz, 28.0, 0.0, 3.0)
                            er, eg, eb = 0.55 + 0.9 * lp, 0.3 + 0.5 * lp, 0.12 + 0.2 * lp
                        elif abs(ua - 0.5 * wa) < 1.2:
                            cr, cg, cb = 0.1, 0.1, 0.13      # median
                        else:
                            inb = ua < 0.5 * wa
                            lane = (ua - 3.5) / 3.3 if inb else (ua - 0.5 * wa - 1.2) / 3.3
                            li_ = int(lane)
                            dl = abs(lane - li_ - 0.5)
                            tr = _sstep(0.32 + fp * 0.5, 0.08, dl)
                            den = 0.55 + 0.45 * math.sin(zg * 0.011 + li_ * 1.7 + (0.0 if inb else 2.0)) *                                 math.sin(zg * 0.0037 + li_ * 0.9)
                            den = max(den, 0.15)
                            if inb:
                                er, eg, eb = 1.5 * tr * den, 1.25 * tr * den, 0.85 * tr * den
                            else:
                                er, eg, eb = 1.4 * tr * den, 0.16 * tr * den, 0.08 * tr * den
                            # lane markings
                            mk = _pulse(zg, fz, 12.0, 0.0, 6.0) * _sstep(0.12 + fp, 0.02, abs(lane - li_))
                            cr += 0.25 * mk
                            cg += 0.24 * mk
                            cb += 0.26 * mk
                        li = 1.0
                    eon = -10.0
                    ew = li
                    zh = zg
                    hit = True
            # ---- buildings, painter's order far -> near
            for li_ in range(cnt[tile], cnt[tile + 1]):
                i = lst[li_]
                if sx < bb[i, 0] or sx > bb[i, 2]:
                    continue
                x0, x1, z0, z1, hb = B[i, BX0], B[i, BX1], B[i, BZ0], B[i, BZ1], B[i, BH]
                base = B[i, BASE]
                mat = int(B[i, MAT])
                face = -1
                zz = 0.0
                u = 0.0
                v = 0.0
                fu = 0.0
                fv = 0.0
                Yh = 0.0
                # front face z = z0
                X = dx * z0 / f
                Y = cam_h - dy * z0 / f
                if X >= x0 and X <= x1 and Y >= base and Y <= hb:
                    face = 0
                    zz = z0
                    u = X - x0
                    v = Y - base
                    fu = z0 / f / ss
                    fv = fu
                    Yh = Y
                    if mat == M_FENCE and _fence_gap(u, v, hb - base, fu):
                        face = -1
                if face < 0:
                    xs = 0.0
                    sd = 0
                    if x0 > 0.0 and dx > 0.0:
                        xs = x0
                        sd = 1
                    elif x1 < 0.0 and dx < 0.0:
                        xs = x1
                        sd = 2
                    if sd > 0:
                        zs = f * xs / dx
                        Y = cam_h - dy * zs / f
                        if zs >= z0 and zs <= z1 and Y >= base and Y <= hb:
                            face = sd
                            zz = zs
                            u = zs - z0
                            v = Y - base
                            fv = zs / f / ss
                            fu = fv * zs / abs(xs) if abs(xs) > 1e-3 else 1e3
                            Yh = Y
                            if mat == M_FENCE and _fence_gap(u, v, hb - base, fu):
                                face = -1
                    if face < 0 and mat == M_FENCE:
                        # back face of a fence ring (seen through the front)
                        X = dx * z1 / f
                        Y = cam_h - dy * z1 / f
                        if X >= x0 and X <= x1 and Y >= base and Y <= hb:
                            if not _fence_gap(X - x0, Y - base, hb - base, z1 / f / ss):
                                face = 2
                                zz = z1
                                u = X - x0
                                v = Y - base
                                fu = z1 / f / ss
                                fv = fu
                                Yh = Y
                if face < 0 and hb < cam_h and dy > 0.0 and mat != M_FENCE:
                    zr = f * (cam_h - hb) / dy
                    X = dx * zr / f
                    if zr >= z0 and zr <= z1 and X >= x0 and X <= x1:
                        face = 3
                        zz = zr
                        u = X - x0
                        v = zr - z0
                        Yh = hb
                if face < 0:
                    continue
                hit = True
                zh = zz
                yw_hit = Yh
                if face == 3:
                    # roof: reflects the violet zenith; material by roof type; thin parapet rim, the far
                    # (sunward) parapet catches a pink glint
                    ar, ag, ab = B[i, AR], B[i, AG], B[i, AB]
                    kf = v / max(z1 - z0, 1.0)
                    pw = zz / f / ss * 1.2
                    edge = _sstep(pw * 1.2 + 0.25, 0.0, (z1 - z0) - v)
                    edge2 = max(_sstep(pw * 1.5 + 0.3, 0.0, v), _sstep(pw * 1.5 + 0.3, 0.0, u),
                                _sstep(pw * 1.5 + 0.3, 0.0, (x1 - x0) - u))
                    rt = int(B[i, ROOFT])
                    # round 10: roofs are the lightest planes of the rooftop sea (they mirror the open sky)
                    if rt == 0:     # grey-violet gravel / membrane
                        rr_, rg_, rb_ = 0.3, 0.3, 0.44
                    elif rt == 1:   # pale concrete
                        rr_, rg_, rb_ = 0.44, 0.42, 0.54
                    elif rt == 2:   # green painted
                        rr_, rg_, rb_ = 0.2, 0.32, 0.34
                    elif rt == 3:   # dark
                        rr_, rg_, rb_ = 0.17, 0.17, 0.28
                    else:           # rust / terracotta
                        rr_, rg_, rb_ = 0.4, 0.25, 0.3
                    rv = _hash(int(B[i, SEED]), 7, 7, 7)
                    rr_, rg_, rb_ = rr_ * 1.3, rg_ * 1.3, rb_ * 1.25
                    cr = (rr_ + 0.03 * kf + 0.04 * rv) * (0.7 + 0.3 * ar)
                    cg = (rg_ + 0.02 * kf + 0.03 * rv) * (0.7 + 0.3 * ag)
                    cb = (rb_ + 0.03 * kf + 0.04 * rv) * (0.7 + 0.3 * ab)
                    # parapet lines
                    cr = cr * (1 - 0.35 * edge2) + edge * 0.28
                    cg = cg * (1 - 0.35 * edge2) + edge * 0.14
                    cb = cb * (1 - 0.3 * edge2) + edge * 0.18
                    # helipad on a few large roofs
                    er, eg, eb, eon, ew = 0.0, 0.0, 0.0, 0.0, 0.0
                    if B[i, CROWN] > 1.5:
                        rx = u - 0.5 * (x1 - x0)
                        rz = v - 0.5 * (z1 - z0)
                        rad = 0.36 * min(x1 - x0, z1 - z0)
                        dr = math.sqrt(rx * rx + rz * rz)
                        if abs(dr - rad) < 0.6 + pw:
                            cr, cg, cb = 0.55, 0.5, 0.55
                        elif abs(rx) < rad * 0.4 and (abs(rx) > rad * 0.28 or abs(rz) < 0.5 + pw) and abs(rz) < rad * 0.5:
                            cr, cg, cb = 0.5, 0.48, 0.5
                        if abs(dr - rad - 1.2) < 0.4 + pw and _pulse(math.atan2(rz, rx) * rad, pw, 3.0, 0.0, 0.6) > 0.3:
                            er, eg, eb, eon, ew = 0.3, 1.2, 0.5, -10.0, 1.0
                else:
                    twr = B[i, PAL] > 0.5 and face != 3
                    if twr:
                        _tower(u, v, fu, fv, B[i], face, zz, cam_h, sun_xw, tmp)
                    else:
                        _facade(u, v, fu, fv, B[i], face, zz, tmp, HA, HN, NA, NN)
                    # round 8 (painted LOD): on mid-distance towers a painter does not dot every window.
                    # Where a floor is under ~9 px (1080p) the window grid is averaged across the bays
                    # (lit offices become soft horizontal ribbons, unlit glass becomes one broad plane),
                    # half of the lit ribbons are dropped, and the glass carries a broad reflected-sky
                    # gradient instead of pixel texture.
                    hgt = hb - base
                    simp = 0.0
                    if twr:
                        simp = 0.0
                    elif hgt > 45.0 and mat != M_APT and mat != M_CONC and mat != M_BILL and mat != M_NAME                             and face != 3:
                        ppf = B[i, FLH] * (1920.0 * 2.2) / max(zz, 1.0)
                        simp = _sstep(10.0, 5.5, ppf)
                    elif hgt > 6.0 and mat != M_BILL and mat != M_NAME and face != 3 and zz > 1300.0:
                        # round 9: mid-distance blocks get the painted LOD too (no identical window dots)
                        ppf = B[i, FLH] * (1920.0 * 2.2) / max(zz, 1.0)
                        simp = 0.9 * _sstep(13.0, 6.0, ppf)
                    if simp > 0.0:
                        bw_ = B[i, BAYW]
                        _facade(u, v, max(fu, bw_ * 3.0), fv * (1.0 + 0.8 * simp), B[i], face, zz, tmp2,
                                HA, HN, NA, NN)
                        sd_ = int(B[i, SEED])
                        fl_ = int(math.floor(v / B[i, FLH]))
                        keep = 1.0 if _hash(sd_, fl_ // 2, int(math.floor(u / (bw_ * 9.0))), 311) > 0.5 * simp else 0.0
                        for c_ in range(3):
                            tmp[c_] = tmp[c_] * (1.0 - simp) + tmp2[c_] * simp
                            tmp[3 + c_] = tmp[3 + c_] * (1.0 - simp) + tmp2[3 + c_] * simp * keep * 1.25
                        tmp[7] = tmp[7] * (1.0 - simp) + tmp2[7] * simp * keep
                        if tmp2[6] < tmp[6]:
                            tmp[6] = tmp[6] * (1.0 - simp) + tmp2[6] * simp
                        # broad reflected sky on the glass: pale pink afterglow low, violet zenith high,
                        # plus a large soft reflected-cloud diagonal (one painted gradient per face)
                        if mat == M_GLASS or mat == M_DARK or mat == M_CURTAIN or mat == M_FINS or mat == M_BAND:
                            hv_ = (Y - base) / max(hgt, 1.0)
                            fw_ = (x1 - x0) if face == 0 else (z1 - z0)
                            uu_ = u / max(fw_, 1.0)
                            dg_ = 0.5 + 0.5 * math.sin(3.2 * (uu_ * 0.7 + hv_) + 0.9 * _hash(sd_, 5, 5, 5) * 6.28)
                            gk_ = simp * (0.55 if mat != M_DARK else 0.35)
                            if face == 1:
                                tr_, tg_, tb_ = 0.62 - 0.22 * hv_, 0.34 - 0.08 * hv_, 0.42 + 0.14 * hv_
                            else:
                                tr_, tg_, tb_ = 0.18 + 0.2 * hv_, 0.16 + 0.07 * hv_, 0.32 + 0.16 * hv_
                            tr_ += 0.14 * dg_
                            tg_ += 0.07 * dg_
                            tb_ += 0.08 * dg_
                            tmp[0] = tmp[0] * (1.0 - gk_) + tr_ * gk_
                            tmp[1] = tmp[1] * (1.0 - gk_) + tg_ * gk_
                            tmp[2] = tmp[2] * (1.0 - gk_) + tb_ * gk_
                    cr, cg, cb = tmp[0], tmp[1], tmp[2]
                    if hb - base <= 45.0 and face != 3 and mat != M_BILL and mat != M_NAME:
                        # round 9: painted value breaks between neighbouring mid-rise facades (each block its
                        # own value step) and a lighter top fading to a darker base (one broad gradient)
                        sdv = int(B[i, SEED])
                        vb_ = 0.68 + 0.64 * _hash(sdv, 29, 31, 37)
                        ht_ = _hash(sdv, 41, 43, 47) - 0.5          # warm / cool paint variation
                        hv2 = (Y - base) / max(hb - base, 1.0)
                        vg_ = 0.82 + 0.3 * hv2
                        cr = cr * vb_ * vg_ * (1.0 + 0.3 * ht_)
                        cg = cg * vb_ * vg_
                        cb = cb * vb_ * vg_ * (1.0 - 0.25 * ht_)
                    if face == 0:
                        # sunward vertical edge of the front face catches a sliver of afterglow
                        wdt = max(x1 - x0, 1.0)
                        side = 1.0 if (x0 + x1) * 0.5 > sun_xw * zz else -1.0
                        g = u / wdt if side < 0 else 1.0 - u / wdt
                        gl0 = 0.4 if (mat == M_GLASS or mat == M_DARK or mat == M_CURTAIN) else 0.14
                        if twr:
                            gl0 = 0.0
                        g = g ** 10 * gl0
                        cr += g * 1.2
                        cg += g * 0.6
                        cb += g * 0.4
                        if hb <= 45.0 and mat != M_BILL and mat != M_NAME:
                            # round 9: warm sun-side rim on the mid-rise blocks too (thin corner line + roofline),
                            # strongest toward the sun axis
                            pwr = zz / f
                            dm = u if side > 0 else wdt - u
                            ax = math.exp(-abs((x0 + x1) * 0.5 / zz - sun_xw) * 5.0)
                            rl = _sstep(pwr * 1.7, pwr * 0.5, dm) * (0.1 + 0.6 * ax * ax)
                            rt_ = _sstep(pwr * 1.4 + 0.1, 0.0, hb - Y) * (0.06 + 0.4 * ax * ax)
                            rr_ = max(rl, rt_)
                            cr += rr_ * 1.3
                            cg += rr_ * 0.72
                            cb += rr_ * 0.42
                        if hb > 45.0:
                            # crisp sunward rim line (~1.5 px) down the lit corner of the towers, hottest
                            # high up (white-gold), fading toward the street
                            pwr = zz / f
                            dm = u if side > 0 else wdt - u
                            rl = _sstep(pwr * 1.8, pwr * 0.6, dm) * (0.45 + 0.55 * _sstep(20.0, 120.0, Y))
                            cr += rl * 1.6
                            cg += rl * 1.05
                            cb += rl * 0.6
                            # round 8: a sun-glint streak - the afterglow mirrored in the curtain wall as one
                            # narrow slanted sheet of light (crisp core, soft shoulder), on about half the
                            # glass towers, hottest high up
                            sdz = int(B[i, SEED])
                            if (not twr) and (mat == M_GLASS or mat == M_CURTAIN or mat == M_BAND or mat == M_DARK) and                                     _hash(sdz, 13, 17, 19) < 0.45 and (hb - base) > 2.2 * wdt:
                                hvz = (Y - base) / max(hb - base, 1.0)
                                gu = u / wdt if side < 0 else 1.0 - u / wdt
                                c0 = 0.18 + 0.3 * _hash(sdz, 13, 17, 23)
                                dgz = gu - c0 - 0.6 * (hvz - 0.75)
                                pxu = pwr / wdt
                                core_z = _sstep(0.05 + 1.2 * pxu, 0.02, abs(dgz))
                                sh_z = math.exp(-(dgz / 0.12) ** 2)
                                gz = (0.9 * core_z + 0.25 * sh_z) * _sstep(0.25, 0.8, hvz) *                                     (0.6 if mat == M_DARK else 1.0)
                                cr += gz * 1.1
                                cg += gz * 0.72
                                cb += gz * 0.55
                                tmp[3] += gz * 0.6
                                tmp[4] += gz * 0.36
                                tmp[5] += gz * 0.22
                                if gz > 0.05:
                                    tmp[6] = min(tmp[6], -10.0)
                                    tmp[7] = max(tmp[7], gz)
                    er, eg, eb = tmp[3], tmp[4], tmp[5]
                    eon = tmp[6]
                    ew = tmp[7]
                    pw = zz / f / ss
                    tl = _sstep(pw * 2.0 + 0.4, 0.0, hb - Y)
                    if face == 1:
                        cr += tl * 0.5
                        cg += tl * 0.28
                        cb += tl * 0.14
                        # hot specular sliver on the far (sunward) vertical corner of the lit side
                        ce = _sstep(pw * 2.5 * zz / max(abs(x0), 1.0) + 0.6, 0.0, z1 - zz)
                        gl = 1.0 if (mat == M_GLASS or mat == M_DARK) else 0.45
                        cr += ce * 1.1 * gl
                        cg += ce * 0.7 * gl
                        cb += ce * 0.4 * gl
                        # round 6: bright edge line on the near (west) corner where the lit face turns away
                        if hb > 45.0:
                            fpx = zz / f * zz / max(abs(x0), 1.0)
                            rl2 = _sstep(fpx * 1.6, fpx * 0.4, u) * (0.5 + 0.5 * _sstep(20.0, 150.0, Y))
                            cr += rl2 * 1.5
                            cg += rl2 * 1.05
                            cb += rl2 * 0.6
                            er += rl2 * 0.5
                            eg += rl2 * 0.35
                            eb += rl2 * 0.2
                            if eon > -5.0 and rl2 > 0.3:
                                eon = -10.0
                            ew = max(ew, rl2 * 0.5)
                        # round 5: warm sun glint on west-facing glass - a soft diagonal band of the afterglow
                        # mirrored across the curtain wall, hottest high up, as HDR emission (blooms)
                        if (not twr) and (mat == M_GLASS or mat == M_DARK or mat == M_CURTAIN or mat == M_FINS) and hb > 45.0:
                            dpth = max(z1 - z0, 1.0)
                            q = (zz - z0) / dpth
                            hv = (Y - base) / max(hb - base, 1.0)
                            dg = q * 0.8 - hv + 0.55 + 0.25 * _hash(int(B[i, SEED]), 3, 1, 9)
                            band = math.exp(-(dg / 0.13) ** 2) * _sstep(0.2, 0.75, hv)
                            gk = band * (0.75 if mat != M_FINS else 0.4)
                            if gk > 0.01:
                                cr += gk * 0.9
                                cg += gk * 0.5
                                cb += gk * 0.25
                                er += gk * 1.3
                                eg += gk * 0.72
                                eb += gk * 0.32
                                if eon > -5.0:
                                    eon = -10.0
                                ew = max(ew, gk)
                                # round 6: hard white-gold sun glints on individual panes inside the band
                                flg = B[i, FLH]
                                byg = B[i, BAYW]
                                sdg = int(B[i, SEED])
                                hg = _hash(sdg, int(math.floor((Y - base) / flg)), int(math.floor(u / byg)) // 2, 97)
                                if hg < 0.45 * gk:
                                    pmk = _pulse(u - byg * 0.1, fu, byg, 0.0, byg * 0.8) *                                         _pulse(Y - base - flg * 0.2, fv, flg, 0.0, flg * 0.62)
                                    cr += pmk * 1.2
                                    cg += pmk * 0.95
                                    cb += pmk * 0.6
                                    er += pmk * 2.4
                                    eg += pmk * 1.9
                                    eb += pmk * 1.15
                                    ew = max(ew, pmk)
                    else:
                        # the parapet top catches the open sky: a pale cool line along each roofline
                        cr += tl * 0.3
                        cg += tl * 0.28
                        cb += tl * 0.34
                    # vertical sign strips with glyphs
                    if face == 0 and B[i, SIGN] >= 1.0:
                        if _sign(u, v, fu, B[i], x0, x1, hb - base, tmp, VA, VN):
                            cr, cg, cb = tmp[0], tmp[1], tmp[2]
                            er, eg, eb = tmp[3], tmp[4], tmp[5]
                            eon = tmp[6]
                            ew = tmp[7]
            if not hit:
                continue
            # ---- atmospheric perspective: graded haze toward the local sky colour + warm ground glow
            hxp = int(min(max(sx, 0.0), Wh - 1.0))
            hyp = int(min(max(sy, 0.0), Hh - 1.0))
            hr, hg, hb2 = haze[hyp, hxp, 0], haze[hyp, hxp, 1], haze[hyp, hxp, 2]
            # the air in front of the backlit towers is a cool violet veil except near the afterglow
            ks = math.exp(-abs(sx - (cx + sun_xw * f)) / (0.18 * Wh)) * 0.8
            cm = 0.55 * (1.0 - ks)
            yw = max(yw_hit, 0.0)
            dd = max(zh - 700.0, 0.0) / fogZ
            # the veil gets lighter / bluer with distance (aerial perspective)
            dk = min(max((zh - 2000.0) / 5000.0, 0.0), 1.0)
            # round 6: the veil is a pink-lavender dusk haze that pales with distance (aerial perspective),
            # thicker low down between the mid-rise blocks
            hr = hr * (1 - cm) + (0.4 + 0.2 * dk) * cm
            hg = hg * (1 - cm) + (0.3 + 0.14 * dk) * cm
            hb2 = hb2 * (1 - cm) + (0.56 + 0.14 * dk) * cm
            fa = 1.0 - math.exp(-(0.1 * dd + 0.8 * dd ** 1.6 + 1.3 * dd * math.exp(-yw / 65.0)))
            c0 = cr * (1 - fa) + hr * fa
            c1 = cg * (1 - fa) + hg * fa
            c2 = cb * (1 - fa) + hb2 * fa
            # sodium ground haze: warm light rising out of the street canyons
            gs = math.exp(-yw / 11.0) * (0.12 + 0.4 * min(dd * 2.0, 1.0))
            c0 = c0 * (1 - gs) + sod[0] * gs
            c1 = c1 * (1 - gs) + sod[1] * gs
            c2 = c2 * (1 - gs) + sod[2] * gs
            rgb[py, px, 0] = c0
            rgb[py, px, 1] = c1
            rgb[py, px, 2] = c2
            al[py, px] = 1.0
            iz[py, px] = 1.0 / max(zh, 1.0)
            le = math.exp(-zh / lightZ) * (1 - 0.6 * fa)
            E[py, px, 0] = er * le
            E[py, px, 1] = eg * le
            E[py, px, 2] = eb * le
            Eon[py, px] = eon * (0.3 * er + 0.55 * eg + 0.15 * eb) * le
    return rgb, al, E, Eon, iz


@njit(cache=True, inline='always')
def _lit(on, t):
    d = t - on
    if d < 0.0:
        return 0.0
    if d >= 0.7:
        return 1.0
    # smooth eased switch-on (no stutter: the montage must be flicker-free)
    x = d / 0.7
    return x * x * (3.0 - 2.0 * x)


@njit(cache=True, parallel=True)
def composite_band(img, em, occ, pm, E, on, top, shift, t):
    """Composite a band plate (premultiplied RGBA pm, emission E, on-times) into the frame buffers in place,
    translated horizontally by `shift` px (sub-pixel, linear). Frame x = plate x + shift."""
    h, wp = pm.shape[0], pm.shape[1]
    H, W = img.shape[0], img.shape[1]
    for yy in prange(h):
        y = top + yy
        if y < 0 or y >= H:
            continue
        for x in range(W):
            sx = x - shift
            x0 = int(math.floor(sx))
            fx = sx - x0
            if x0 < 0 or x0 + 1 >= wp:
                continue
            a = pm[yy, x0, 3] * (1 - fx) + pm[yy, x0 + 1, 3] * fx
            if a <= 1e-5:
                continue
            l0 = _lit(on[yy, x0], t)
            l1 = _lit(on[yy, x0 + 1], t)
            for c in range(3):
                col = pm[yy, x0, c] * (1 - fx) + pm[yy, x0 + 1, c] * fx
                e = E[yy, x0, c] * l0 * (1 - fx) + E[yy, x0 + 1, c] * l1 * fx
                img[y, x, c] = img[y, x, c] * (1 - a) + col
                em[y, x, c] = em[y, x, c] * (1 - a) + e
            if a > occ[y, x]:
                occ[y, x] = a
