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
 GRP, SIGN, RUNF, ROOFT, CROWN, PAL, RK, RS) = range(27)
NB = 27
R_TX = 24.0     # texels per text line / char cell in the text atlases (s03_city_dusk_text.R)

# materials
M_CONC, M_GLASS, M_APT, M_BILL, M_FINS, M_BAND, M_DARK, M_FENCE, M_PLAIN, M_TANK, M_CURTAIN, M_NAME, M_ROAD = range(13)


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


@njit(cache=True, inline='always')
def _weather(u, v, hb, fu, seed):
    """round 14: painted value variation inside a facade (multiplier ~0.8..1.15): rain / grime streaks running
    down from the parapet and from some window sills (vertical, soft-ended, varied per column), broad painted
    light / dark patches, a lighter coping band along the top floor. Faded out where the pixel footprint is
    wider than a streak (no speckle at a distance)."""
    k = _sstep(2.4, 0.5, fu)
    cw = 1.3 + 0.9 * _hash(seed, 3, 5, 700)
    c = int(math.floor(u / cw))
    sv = _hash(seed, c, 0, 701)
    L = 4.0 + 26.0 * _hash(seed, c, 1, 702)
    depth = hb - v
    st = 0.0
    if sv < 0.42:
        st = (0.07 + 0.12 * _hash(seed, c, 2, 703)) * math.exp(-max(depth, 0.0) / L)
        # soft streak edges (a painted stroke, not a hard stripe)
        fr = u / cw - c
        st *= _sstep(0.0, 0.3, fr) * _sstep(1.0, 0.7, fr)
    # a second, fainter set of short streaks under random floors
    c2 = int(math.floor(u / (cw * 0.7)))
    f2 = int(math.floor(v / 11.0))
    if _hash(seed, c2, f2, 704) < 0.18:
        st += 0.06 * _sstep(11.0, 2.0, v - f2 * 11.0) * _sstep(11.0, 8.0, v - f2 * 11.0 + 3.0)
    m = 1.0 - st * k
    # broad painted patches (low frequency, always on)
    m *= 1.0 + 0.055 * math.sin(u * 0.11 + seed * 0.7) * math.sin(v * 0.045 + seed * 0.31)         + 0.035 * math.sin(u * 0.037 - v * 0.021 + seed)
    # lighter coping / top-floor band catching the sky
    m *= 1.0 + 0.14 * _sstep(4.5, 3.0, depth) * _sstep(0.0, 0.4, depth)
    return m


@njit(cache=True, inline='always')
def _slant_depth(rs, z0, z1, hb, base):
    """depth (m) of the street-slant roof plane of a RK=1 building"""
    return min(0.45 * (z1 - z0), 0.35 * (hb - base) / max(rs, 0.05))


@njit(cache=True, inline='always')
def _roof_top(rk, rs, x0, x1, z0, z1, hb, base, X, Z):
    """round 11: local roof height of a pitched / slanted roof at (X, Z)"""
    if rk == 1:
        dsl = _slant_depth(rs, z0, z1, hb, base)
        return hb - rs * max(z0 + dsl - Z, 0.0)
    if rk == 2:
        return hb - rs * abs(Z - 0.5 * (z0 + z1))
    return hb - rs * abs(X - 0.5 * (x0 + x1))


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
    if face == 0:      # facing camera (east): cool sky ambient (round 13: more neutral, materials read)
        lr, lg, lb = 0.23 + 0.05 * hv, 0.22 + 0.06 * hv, 0.245 + 0.08 * hv
        if mat == M_CONC or mat == M_APT or mat == M_PLAIN or mat == M_TANK:
            # round 13: real material colours (beige tile, grey concrete, brown brick, white panel) carry
            # through the shade: push the albedo away from its grey
            am_ = (ar + ag + ab) / 3.0
            ar, ag, ab = am_ + (ar - am_) * 1.7, am_ + (ag - am_) * 1.7, am_ + (ab - am_) * 1.7
    elif face == 1:    # facing the afterglow: grazing gold sun, the lower floors in the neighbours' shade
        ks_ = _sstep(0.0, 1.0, v / max(B[BH] - B[BASE], 1.0)) ** 0.7
        lr = 0.3 + 0.62 * ks_ + 0.2 * hv
        lg = 0.2 + 0.34 * ks_ + 0.08 * hv
        lb = 0.24 + 0.1 * ks_ + 0.03 * hv
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
                    leg = _sstep(5.0, 8.5, lh / max(2.0 * fu, 1e-6))
                    g2 = g2 * leg + 0.2 * (1.0 - leg)
                    m2 = max(m2, g2 * (1.0 if ln == 0 else 0.9))
        cr, cg, cb = br * (1 - m2) + ar2 * m2, bg * (1 - m2) + ag2 * m2, bb2 * (1 - m2) + ab2 * m2
        lit = B[LIT0]
        out[0], out[1], out[2] = cr * 0.35, cg * 0.3, cb * 0.35
        if lit > 0.5:
            li = 0.5 * (0.9 + 0.2 * vv)
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
        # round 14: glyphs under ~8 output px are not drawn as (pseudo-)text: the board melts into its lit
        # panel colour (bands are rendered at 2x: fu is half an output pixel)
        leg = _sstep(5.0, 8.5, sc * R_TX * 0.78 / max(2.0 * fu, 1e-6))
        g = g * leg + 0.2 * (1.0 - leg)
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
    if mat == M_ROAD:
        # round 12: elevated expressway girder: dark concrete box girder, a pale sound-barrier panel band
        # along the top, a thin line of light under the deck lip
        hbr = hb
        bar = _sstep(hbr * 0.55 - fv, hbr * 0.55 + fv, v)
        lip = _pulse(v - hbr * 0.5, fv, 1e6, 0.0, 0.25)
        k_ = 0.75 + 0.75 * bar
        out[0], out[1], out[2] = wr * k_ + 0.06 * lip, wg * k_ + 0.05 * lip, wb * k_ + 0.05 * lip
        if face == 0 or face == 1:
            pj = _pulse(u, fu, 2.0, 0.0, 0.12) * bar
            out[0], out[1], out[2] = out[0] * (1 - 0.25 * pj), out[1] * (1 - 0.25 * pj), out[2] * (1 - 0.2 * pj)
        out[3], out[4], out[5] = 0.5 * lip, 0.32 * lip, 0.15 * lip
        out[6] = -10.0
        out[7] = lip
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
            # round 12: the vertical specular streak only on a few curtain walls (it was on every one)
            kst = 1.0 if _hash(seed, 4, 4, 4) < 0.25 else 0.0
            st = math.exp(-((uu - sp0) / sw_) ** 2) * (0.35 + 0.65 * q) * kst
            st2 = math.exp(-((uu - sp0 - 0.16) / 0.012) ** 2) * (0.3 + 0.7 * q) * kst
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
            # round 13: lighter, sky-coloured and varied per building (teal / neutral / warm tinted glass)
            if q < 0.5:
                gr, gg, gb = _lerp3(0.14, 0.15, 0.2, 0.2, 0.25, 0.36, q / 0.5)
            elif q < 0.72:
                gr, gg, gb = _lerp3(0.2, 0.25, 0.36, 0.42, 0.32, 0.44, (q - 0.5) / 0.22)
            else:
                gr, gg, gb = _lerp3(0.42, 0.32, 0.44, 0.36, 0.38, 0.54, (q - 0.72) / 0.28)
            ht3 = _hash(seed, 21, 23, 29)
            if ht3 < 0.35:
                gr, gg, gb = gr * 0.8, gg * 1.08, gb * 1.05
            elif ht3 < 0.6:
                gr, gg, gb = gr * 1.2, gg * 1.02, gb * 0.8
            vk3 = 0.75 + 0.5 * _hash(seed, 31, 37, 41)
            gr, gg, gb = gr * vk3, gg * vk3, gb * vk3
        else:
            k = _sstep(10.0, 120.0, vb)
            hsk = _hash(seed, 13, 1, 3)
            gr, gg, gb = 0.11 + 0.1 * k + 0.05 * hsk, 0.12 + 0.1 * k + 0.04 * hsk, 0.22 + 0.12 * k
            # round 13: not every pane mirrors the blue sky - drawn curtains / blinds (pale warm greys),
            # frosted glass, darker open panes; varied per window (whole floors of one tenant often match)
            hw_ = _hash(seed, int(math.floor(v / B[FLH])), int(math.floor(u / (B[BAYW] * 2.0))), 613)
            if hw_ < 0.3:
                gr, gg, gb = 0.3 + 0.08 * hsk, 0.27 + 0.06 * hsk, 0.25
            elif hw_ < 0.42:
                gr, gg, gb = 0.24, 0.26, 0.28
            elif hw_ < 0.55:
                gr, gg, gb = 0.06, 0.07, 0.11
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
    if face != 2 and mat != M_TANK:
        wz_ = _weather(u, v, hb, fu, seed)
        wr, wg, wb = wr * wz_, wg * wz_, wb * wz_ * (0.97 + 0.03 * wz_)
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
    leg = _sstep(5.0, 8.5, cell / max(2.0 * fu, 1e-6))      # round 14: no pseudo-text at tiny sizes
    g = g * leg + 0.2 * (1.0 - leg)
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
# pier w (m), corner return w (bays), punched (0 curtain / 1 punched precast / stone)
PALS = np.array([
    # frame               tint                refl  spf   mul   np   pw    cw   punched style
    [0.30, 0.36, 0.48, 0.55, 0.72, 1.00, 0.92, 0.14, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0],   # 0 blue curtain wall
    [0.86, 0.87, 0.92, 0.52, 0.62, 0.82, 0.70, 0.30, 0.30, 0.0, 0.0, 1.0, 0.0, 0.0],   # 1 white aluminium grid
    [0.26, 0.38, 0.40, 0.42, 0.86, 0.84, 0.90, 0.16, 0.14, 4.0, 0.5, 0.0, 0.0, 0.0],   # 2 teal-green glass
    [0.80, 0.76, 0.70, 0.50, 0.58, 0.70, 0.50, 0.50, 0.60, 3.0, 0.7, 1.0, 1.0, 0.0],   # 3 precast concrete, punched
    [0.11, 0.11, 0.14, 0.36, 0.38, 0.50, 0.85, 0.12, 0.10, 0.0, 0.0, 0.5, 0.0, 0.0],   # 4 dark smoked glass
    [0.34, 0.26, 0.22, 0.92, 0.70, 0.52, 0.88, 0.18, 0.14, 6.0, 0.4, 0.0, 0.0, 0.0],   # 5 bronze glass
    [0.64, 0.66, 0.72, 0.52, 0.64, 0.86, 0.80, 0.12, 0.42, 1.0, 0.42, 1.0, 0.0, 0.0],  # 6 silver vertical fins
    [0.74, 0.62, 0.58, 0.56, 0.60, 0.78, 0.62, 0.52, 0.45, 2.0, 0.5, 1.0, 1.0, 0.0],   # 7 warm rose stone, punched
    [0.72, 0.78, 0.88, 0.70, 0.88, 1.06, 0.95, 0.12, 0.12, 0.0, 0.0, 0.5, 0.0, 0.0],   # 8 pale blue curtain wall
    [1.00, 0.80, 0.56, 0.52, 0.56, 0.70, 0.45, 0.50, 0.60, 2.0, 0.8, 1.0, 1.0, 0.0],   # 9 warm tan / beige stone
    [1.00, 1.00, 1.00, 0.46, 0.52, 0.66, 0.40, 0.50, 0.50, 0.0, 0.0, 1.0, 0.0, 1.0],   # 10 white-panel residential
    [0.10, 0.19, 0.22, 0.46, 0.80, 0.86, 0.96, 0.10, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0],   # 11 dark blue-green glass
    [0.80, 0.40, 0.28, 0.50, 0.52, 0.66, 0.42, 0.50, 0.55, 2.0, 0.7, 1.0, 1.0, 0.0],   # 12 warm red-brown brick, punched
])
NPAL = 13


@njit(cache=True)
def _tower(u, v, fu, fv, B, face, z, cam_h, sunx, out):
    """Round 11: painted high-rise facade. u, v face coords (m), fu, fv pixel footprint (m).
    Lighting logic for a low sun straight ahead: the camera-facing face (0) is in shadow and mirrors the cool
    eastern sky (lilac high -> dusk blue -> a dusty warm band low); the -x face (1) is grazed by the sun and
    mirrors hot gold (gold low -> apricot -> rose-lilac high); towers close to the sun in screen space are
    backlit silhouettes (dark, desaturated, cool). Structure: continuous vertical mullions, thin spandrel bands,
    piers / corner returns / mechanical floors, window-cleaning gondola rails under the parapet, a warm stone
    lobby. Writes out[0:3] colour, out[3:6] emission, out[6] on-time, out[7] emission weight."""
    seed = int(B[SEED])
    pal = ((int(B[PAL]) - 1) % 16) % NPAL
    P = PALS[pal]
    hb = B[BH] - B[BASE]
    vb = v + B[BASE]
    wdt = (B[BX1] - B[BX0]) if face != 1 and face != 2 else (B[BZ1] - B[BZ0])
    wdt = max(wdt, 1.0)
    flh = B[FLH]
    bay = B[BAYW]
    punched = P[12] > 0.5
    if punched:
        bay = max(bay, 2.6)
    tseed = int(B[PAL]) // 16                        # tower id: shared by all tiers of one tower
    xc = 0.5 * (B[BX0] + B[BX1])
    zc = max(0.5 * (B[BZ0] + B[BZ1]), 1.0)
    ds = abs(xc / zc - sunx) * 2.2                   # screen distance to the sun (fraction of W)
    # round 13: towers left of the sun turn their +x face toward the afterglow: paint it as the lit face
    lside = xc / zc < sunx
    face0 = face
    if face == 2 and lside:
        face = 1
    bl = _sstep(0.34, 0.12, ds)                      # 1 = backlit silhouette
    hv = min(max(vb / 300.0, 0.0), 1.0)
    q = min(max(v / max(hb, 1.0), 0.0), 1.0)
    uu = min(max(u / wdt, 0.0), 1.0)
    # ---- reflected sky (painted)
    if face == 1:
        if hv < 0.5:
            sr, sg, sb = _lerp3(1.5, 0.92, 0.42, 1.28, 0.62, 0.4, hv / 0.5)
        else:
            sr, sg, sb = _lerp3(1.28, 0.62, 0.4, 0.82, 0.5, 0.68, (hv - 0.5) / 0.5)
        # hottest toward the sunward (far) edge
        kx = 0.72 + 0.5 * uu
        sr, sg, sb = sr * kx, sg * kx, sb * (0.85 + 0.25 * uu)
    elif face == 0:
        # round 13: sky-coloured vertical glass gradient on every tower: warm dusty apricot at the base
        # (the lit haze over the city), violet-blue mid, pink-lilac at the top (the upper dusk sky)
        if hv < 0.3:
            sr, sg, sb = _lerp3(0.62, 0.38, 0.32, 0.3, 0.27, 0.46, hv / 0.3)
        elif hv < 0.7:
            sr, sg, sb = _lerp3(0.3, 0.27, 0.46, 0.46, 0.36, 0.62, (hv - 0.3) / 0.4)
        else:
            sr, sg, sb = _lerp3(0.46, 0.36, 0.62, 0.7, 0.5, 0.74, (hv - 0.7) / 0.3)
        tv = 0.8 + 0.4 * _hash(tseed, 5, 7, 423)          # per-tower value
        sr, sg, sb = sr * tv, sg * tv, sb * tv
    else:
        sr, sg, sb = 0.07, 0.08, 0.16
    # one broad painted sheen across the face (not a flat tint)
    sh = 0.5 + 0.5 * math.sin(3.0 * (uu * 0.8 + q * 1.3) + tseed * 1.7)
    sk_ = 0.88 + 0.24 * sh
    sr, sg, sb = sr * sk_, sg * sk_, sb * sk_
    # per mirror panel tilt: the reflection steps from panel to panel
    pf = 2 + (tseed % 2)
    pb = 1 + (tseed // 3) % 3
    pfl = math.floor(v / (flh * pf))
    pcl = math.floor(u / (bay * pb))
    jamp = _sstep(bay * pb * 1.2, bay * pb * 0.4, fu)
    jit = (_hash(seed, int(pfl), int(pcl), 401) - 0.5) * 0.07 * jamp
    sr, sg, sb = sr * (1 + jit), sg * (1 + jit), sb * (1 + jit * 0.7)
    # round 12: painted sky / cumulus reflections on a few hero towers (RS = 1 + variant). Each tower mirrors a
    # different part of the sky: a peach cumulus flank, the violet-blue upper sky, gold afterglow, teal ->
    # peach; each with its own gradient direction, broken per mirror panel, with big soft cloud lobes.
    if B[RS] > 0.5 and face != 2:
        var = int(B[RS] - 1.0 + 0.5) % 4
        off = (_hash(seed, int(pfl), int(pcl), 411) - 0.5) * jamp
        ang = (0.1 + 0.8 * _hash(tseed, 12, 5, 431)) * math.pi
        gc = math.cos(ang) * (uu - 0.5) * 0.6 + math.sin(ang) * (0.5 - q) + 0.5 + 0.04 * off
        if var == 0:
            ar_, ag_, ab_, br_, bg_, bb_ = 1.3, 0.8, 0.62, 0.5, 0.44, 0.82
            cr_, cg_, cb_ = 1.55, 1.08, 0.86
        elif var == 1:
            ar_, ag_, ab_, br_, bg_, bb_ = 0.3, 0.38, 0.92, 0.72, 0.84, 1.1
            cr_, cg_, cb_ = 0.95, 0.98, 1.2
        elif var == 2:
            ar_, ag_, ab_, br_, bg_, bb_ = 1.45, 0.92, 0.5, 0.9, 0.46, 0.62
            cr_, cg_, cb_ = 1.6, 1.2, 0.75
        else:
            ar_, ag_, ab_, br_, bg_, bb_ = 0.24, 0.55, 0.7, 1.2, 0.76, 0.64
            cr_, cg_, cb_ = 1.4, 1.02, 0.86
        tg = _sstep(0.15, 0.9, gc)
        rr_ = ar_ + (br_ - ar_) * tg
        rg_ = ag_ + (bg_ - ag_) * tg
        rb_ = ab_ + (bb_ - ab_) * tg
        # big soft cumulus lobes crossing the face along the gradient direction (firm upper edge)
        dg = (math.cos(ang + 1.2) * u * 0.8 + math.sin(ang + 1.2) * vb) + off * 5.0
        cfld = 0.5 + 0.3 * math.sin(dg * 0.045 + tseed * 1.3) + 0.2 * math.sin(dg * 0.11 - u * 0.03 + tseed)
        cm_ = _sstep(0.58, 0.66, cfld) * _sstep(0.1, 0.4, q)
        km = 0.82 if face == 0 else 0.62
        # round 14: every variant carries the dusk sky's own vertical gradient too: lilac high, peach low
        if q > 0.55:
            lr2, lg2, lb2 = _lerp3(1.0, 0.72, 0.9, 0.78, 0.66, 1.08, (q - 0.55) / 0.45)
        else:
            lr2, lg2, lb2 = _lerp3(1.45, 0.86, 0.6, 1.0, 0.72, 0.9, q / 0.55)
        rr_ = rr_ * 0.55 + lr2 * 0.45
        rg_ = rg_ * 0.55 + lg2 * 0.45
        rb_ = rb_ * 0.55 + lb2 * 0.45
        sr = sr * (1 - km) + rr_ * km
        sg = sg * (1 - km) + rg_ * km
        sb = sb * (1 - km) + rb_ * km
        sr = sr * (1 - 0.6 * cm_) + cr_ * 0.6 * cm_
        sg = sg * (1 - 0.6 * cm_) + cg_ * 0.6 * cm_
        sb = sb * (1 - 0.6 * cm_) + cb_ * 0.6 * cm_
    # the afterglow caught on the sunward part of the shadow-face glass of the side towers (a diagonal gold
    # wedge, as the glass of a slightly angled curtain wall picks up the sun), only on some towers
    if face == 0 and not punched:
        sidep = 1.0 if xc / zc > sunx else -1.0
        us = uu if sidep > 0 else 1.0 - uu
        gs = _hash(tseed, 3, 3, 421)
        gs = 0.0 if gs < 0.8 else 0.35
        wedge = math.exp(-max(us + 0.45 * (1.0 - q) - 0.05, 0.0) / 0.22)
        gold = wedge * _sstep(0.15, 0.75, q) * (1.0 - bl) * gs
        sr = sr * (1 - 0.7 * gold) + 1.5 * 0.7 * gold
        sg = sg * (1 - 0.7 * gold) + 0.92 * 0.7 * gold
        sb = sb * (1 - 0.7 * gold) + 0.5 * 0.7 * gold
    refl = P[6]
    gr_ = sr * P[3] * refl + 0.03 * (1 - refl)
    gg_ = sg * P[4] * refl + 0.035 * (1 - refl)
    gb_ = sb * P[5] * refl + 0.07 * (1 - refl)
    # ---- frame / wall light
    if face == 0:
        # round 12: shade light keeps the material's own hue (stone reads warm, white panels read white)
        tvf = 0.72 + 0.56 * _hash(tseed, 9, 2, 433)
        lr, lg, lb = (0.33 + 0.08 * hv) * tvf, (0.32 + 0.08 * hv) * tvf, (0.42 + 0.1 * hv) * tvf
    elif face == 1:
        lr = (0.95 + 0.35 * hv) * (0.7 + 0.45 * uu)
        lg = (0.6 + 0.12 * hv) * (0.7 + 0.45 * uu)
        lb = 0.42 + 0.06 * hv
    else:
        lr, lg, lb = 0.1, 0.12, 0.24
    ab = (B[AR] + B[AG] + B[AB]) / 3.0
    vk = 0.85 + 0.3 * (ab - 0.85)
    fr_, fg_, fb_ = P[0] * lr * vk, P[1] * lg * vk, P[2] * lb * vk
    # ---- structure masks
    spf = P[7]
    mw = P[8]
    fl = math.floor(v / flh)
    ci = math.floor(u / bay)
    if punched:
        # precast / stone: punched windows ~half the bay, deep reveals
        # tall narrow windows stacked in continuous vertical strips (the precast piers read as lines)
        pu = _pulse(u - bay * 0.3, fu, bay, 0.0, bay * 0.42)
        pv = _pulse(v - flh * 0.12, fv, flh, 0.0, flh * 0.76)
    else:
        pu = _pulse(u - 0.5 * mw, fu, bay, 0.0, bay - mw)
        pv = _pulse(v - spf * flh * 0.6, fv, flh, 0.0, flh * (1.0 - spf))
    gm = pu * pv
    npb = int(P[9])
    pier = 0.0
    if npb > 0:
        pier = _pulse(u + 0.5 * P[10], fu, bay * npb, 0.0, P[10])
    cw = P[11] * bay
    corner = 0.0
    if cw > 0.0:
        corner = max(_sstep(cw + fu, cw, u), _sstep(wdt - cw - fu, wdt - cw, u))
    mf = 10 + tseed % 7
    mech = 1.0 if (int(fl) % mf) == mf - 1 and v > flh * 3 and v < hb - flh * 2 else 0.0
    par = _sstep(hb - 1.8 - fv, hb - 1.8, v)
    lob = 1.0 - _sstep(0.0, 0.5, vb - 9.0)
    resid = P[13] > 0.5
    if resid:
        # white-panel residential: continuous balcony parapets (pale bands), recessed dark glass, unit
        # partitions every two bays
        pv = 1.0 - _pulse(v, fv, flh, 0.0, flh * 0.4)
        pu = 1.0 - _pulse(u, fu, bay * 2.0, 0.0, 0.45)
        gm = pv * pu
        mech = 0.0
    solid = max(max(pier, corner), max(par, mech))
    gm = gm * (1.0 - solid)
    # ---- compose: glass, spandrel (opaque glass or wall), mullions (frame)
    if resid:
        gr_, gg_, gb_ = gr_ * 0.55 + 0.02, gg_ * 0.55 + 0.02, gb_ * 0.6 + 0.04
        cr = gr_ * gm + fr_ * 1.12 * (1 - gm)
        cg = gg_ * gm + fg_ * 1.12 * (1 - gm)
        cb = gb_ * gm + fb_ * 1.1 * (1 - gm)
        solid = 0.0
    elif punched:
        # window glass sits in a shadowed reveal: dimmer reflection, darker top edge
        rv = _sstep(0.0, 0.3, (v - fl * flh - flh * 0.12) / max(flh * 0.76, 0.1))
        gr_, gg_, gb_ = gr_ * (0.6 + 0.3 * rv), gg_ * (0.6 + 0.3 * rv), gb_ * (0.65 + 0.3 * rv)
        wr_, wg_, wb_ = fr_, fg_, fb_
        # at a distance the punched grid melts into the wall (painted LOD: no dot grid)
        mlt = _sstep(bay * 0.9, bay * 0.3, fu) * 0.55 + 0.45
        gr_, gg_, gb_ = wr_ + (gr_ - wr_) * mlt, wg_ + (gg_ - wg_) * mlt, wb_ + (gb_ - wb_) * mlt
        # horizontal precast joints
        jl = _pulse(v, fv, flh, 0.0, 0.1)
        wr_, wg_, wb_ = wr_ * (1 - 0.2 * jl), wg_ * (1 - 0.2 * jl), wb_ * (1 - 0.18 * jl)
        cr = gr_ * gm + wr_ * (1 - gm)
        cg = gg_ * gm + wg_ * (1 - gm)
        cb = gb_ * gm + wb_ * (1 - gm)
    else:
        spm = 1.0 - pv
        mulm = max(pv - pu * pv, 0.0)
        # spandrels: opaque back-painted glass, a little darker and flatter than the vision glass
        spr, spg, spb = gr_ * 0.7 + fr_ * 0.2, gg_ * 0.7 + fg_ * 0.2, gb_ * 0.72 + fb_ * 0.2
        # mullions: continuous vertical lines, lighter than the glass on the shadow face (they catch the sky)
        mk = 1.25 if face == 0 else 0.85
        cr = gr_ * gm + (spr * spm + fr_ * mk * mulm) * (1 - solid)
        cg = gg_ * gm + (spg * spm + fg_ * mk * mulm) * (1 - solid)
        cb = gb_ * gm + (spb * spm + fb_ * mk * mulm) * (1 - solid)
    sk2 = 1.25 if (face == 1 and pier > 0.0) else 1.0
    cr += fr_ * solid * sk2
    cg += fg_ * solid * sk2
    cb += fb_ * solid * sk2
    if mech > 0.0:
        lv = _pulse(v, fv, 0.45, 0.0, 0.15)
        cr, cg, cb = cr * (0.8 - 0.2 * lv), cg * (0.8 - 0.2 * lv), cb * (0.82 - 0.2 * lv)
    # window-cleaning gondola rails under the parapet: two thin lines catching the light
    rails = max(_pulse(v - (hb - 2.6), fv, 1e6, 0.0, 0.25), _pulse(v - (hb - 4.0), fv, 1e6, 0.0, 0.2))
    if rails > 0.0 and hb > 40.0:
        if face == 1:
            cr, cg, cb = cr + 0.5 * rails, cg + 0.3 * rails, cb + 0.16 * rails
        elif face == 0:
            cr, cg, cb = cr + 0.1 * rails, cg + 0.1 * rails, cb + 0.14 * rails
    if lob > 0.0:
        # warm stone lobby base
        cr = cr * (1 - lob) + 0.55 * lr * lob
        cg = cg * (1 - lob) + 0.42 * lg * lob
        cb = cb * (1 - lob) + 0.34 * lb * lob
    # broad painted value gradient: darker toward the base
    vg = 0.74 + 0.36 * _sstep(0.0, 1.0, hv)
    if face != 2:
        # round 14: painted variation - streaks / patches mostly on the opaque parts, a light top band
        wz_ = _weather(u, v, hb, fu, seed)
        wz_ = 1.0 + (wz_ - 1.0) * (0.45 + 0.55 * (1.0 - gm))
        vg *= wz_
    cr, cg, cb = cr * vg, cg * vg, cb * vg
    # ---- backlit towers (near the sun): dark, desaturated, cool silhouettes; structure barely readable
    if bl > 0.0:
        lum = 0.3 * cr + 0.55 * cg + 0.15 * cb
        if face == 1:
            tr_, tg_, tb_ = 0.2 + 0.25 * lum, 0.13 + 0.15 * lum, 0.15 + 0.12 * lum
        else:
            tr_, tg_, tb_ = 0.07 + 0.2 * lum, 0.08 + 0.2 * lum, 0.13 + 0.24 * lum
        kb = 0.55 * bl
        cr, cg, cb = cr * (1 - kb) + tr_ * kb, cg * (1 - kb) + tg_ * kb, cb * (1 - kb) + tb_ * kb
    # ---- sun-side silhouette edge: a crisp gold line on the far corner of the lit face (side towers)
    if face == 1:
        rim = _sstep(fu * 2.0, fu * 0.6, wdt - u) * (0.35 + 0.65 * hv) * (1.0 - 0.4 * bl)
        if face0 == 2:
            rim = min(rim * 1.4, 1.0)
        cr, cg, cb = cr + 1.6 * rim, cg + 1.02 * rim, cb + 0.55 * rim
    elif face == 2:
        rim = _sstep(fu * 1.8, fu * 0.6, wdt - u) * (0.3 + 0.5 * hv) * (1.0 - bl)
        cr, cg, cb = cr + 1.0 * rim, cg + 0.62 * rim, cb + 0.36 * rim
    elif face == 0:
        # round 13: a lost, soft edge on the shadow side of the shaded face (it melts into the haze), and a
        # thin hot rim on the sun-side edge where no side face is visible
        ush = u if lside else wdt - u
        lost = _sstep(fu * 7.0, 0.0, ush) * 0.45 * (0.5 + 0.5 * hv)
        cr, cg, cb = cr * (1 - lost) + 0.46 * lost, cg * (1 - lost) + 0.42 * lost, cb * (1 - lost) + 0.6 * lost
        usn = wdt - u if lside else u
        rim0 = _sstep(fu * 2.2, fu * 0.5, usn) * (0.3 + 0.7 * hv) * (1.0 - 0.35 * bl) * 1.1
        cr, cg, cb = cr + 1.5 * rim0, cg + 0.9 * rim0, cb + 0.5 * rim0
    # round 14: a hot gold glint running down the sun-side corner of a few glass towers (wwy_07): tight core,
    # soft falloff along the height, a short bloom
    if B[RS] > 8.5 and face == 0 and bl < 0.9:
        gq = (int(B[RS] - 1.0 + 0.5) // 4 - 2) / 20.0
        usn2 = wdt - u if lside else u
        g1 = _sstep(fu * 3.2 + 0.6, fu * 0.5, usn2) * math.exp(-((q - gq) / 0.16) ** 2)
        g2 = _sstep(fu * 26.0 + 5.0, 0.0, usn2) * math.exp(-((q - gq) / 0.1) ** 2)
        g3 = math.exp(-((q - gq) / 0.035) ** 2) * _sstep(wdt * 0.6, 0.0, usn2)     # a short flare across
        gl = (g1 + 0.95 * g2 + 0.35 * g3) * (1.0 - bl)
        if gl > 0.003:
            k_ = min(gl, 1.0)
            cr = cr * (1 - 0.7 * k_) + 1.2 * k_
            cg = cg * (1 - 0.7 * k_) + 0.66 * k_
            cb = cb * (1 - 0.8 * k_) + 0.2 * k_
            out[0], out[1], out[2] = cr, cg, cb
            out[3], out[4], out[5] = 0.55 * gl, 0.26 * gl, 0.06 * gl
            out[6] = -10.0
            out[7] = min(gl * 2.0, 1.0)
            return
    out[0], out[1], out[2] = cr, cg, cb
    out[3], out[4], out[5], out[6], out[7] = 0.0, 0.0, 0.0, 1e9, 0.0
    if face == 1:
        # the gold glass glows (blooms) where it is hottest
        hot = max(0.3 * cr + 0.55 * cg + 0.15 * cb - 0.5, 0.0) * (1.0 - bl) * max(gm, 0.4)
        if hot > 0.0:
            out[3], out[4], out[5] = cr * hot * 0.9, cg * hot * 0.9, cb * hot * 0.9
            out[6] = -10.0
            out[7] = min(hot * 3.0, 1.0)
        return
    if face == 2 or gm <= 0.0:
        return
    # ---- lights on the shadow face only: whole tenant floors as ribbons (few), none on backlit towers
    grp = 6 + (seed % 7)
    sec = int(math.floor((ci + 7.0 * _hash(seed, int(fl), 0, 409)) / grp))
    run = int(fl)
    r1 = _hash(seed, face + 7, run, sec)
    lit0 = B[LIT0] * 0.35 * (1.0 - bl)
    litadd = B[LITADD] * 0.3 * (1.0 - bl)
    on = 1e9
    if r1 < lit0:
        on = -10.0
    elif r1 < lit0 + litadd:
        clu = run // 4
        on = 0.25 + 4.6 * _hash(seed, clu, sec, 41) + 0.09 * (run - clu * 4)
    if _hash(seed, int(fl), int(ci), 405) < 0.08:
        on = 1e9
    if on > 1e8:
        return
    r3 = _hash(seed, run, sec, 99)
    if r3 < 0.5:
        er, eg, eb = 1.05, 0.9, 0.7
    elif r3 < 0.9:
        er, eg, eb = 0.95, 0.97, 0.92
    else:
        er, eg, eb = 0.62, 0.8, 1.1
    ei = (0.18 + 0.1 * _hash(seed, int(fl), int(ci) // 4, 406)) * (1.0 - 0.4 * refl)
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
                    xa_c = 0.0
                    if street.shape[0] > 8:
                        xa_c = street[6] + street[7] * (zg - 1000.0)
                    if street.shape[0] > 8 and abs(xg - xa_c) <= street[8]:
                        # round 11: a diagonal boulevard receding toward the afterglow - dark asphalt, pale
                        # sidewalks with dotted lamps, and individual cars as tiny light dots (white-gold
                        # headlights inbound, red tail lights outbound), bunched by traffic (static)
                        wa = 2.0 * street[8]
                        ua = xg - (xa_c - street[8])
                        cr, cg, cb = 0.06, 0.06, 0.1
                        er, eg, eb = 0.0, 0.0, 0.0
                        if ua < 3.5 or ua > wa - 3.5:
                            cr, cg, cb = 0.16, 0.13, 0.18
                            lp = _pulse(zg, fz, 31.0, 0.0, 1.2)
                            er, eg, eb = 0.12 + 0.9 * lp, 0.07 + 0.55 * lp, 0.03 + 0.22 * lp
                        elif abs(ua - 0.5 * wa) < 1.2:
                            cr, cg, cb = 0.09, 0.1, 0.12      # median
                        else:
                            inb = ua < 0.5 * wa
                            lane = (ua - 3.5) / 3.3 if inb else (ua - 0.5 * wa - 1.2) / 3.3
                            li_ = int(lane)
                            dl = abs(lane - li_ - 0.5)
                            acr = _sstep(0.3 + fp * 0.3, 0.18, dl)
                            den = 0.5 + 0.5 * math.sin(zg * 0.009 + li_ * 1.7 + (0.0 if inb else 2.0)) *                                 math.sin(zg * 0.0031 + li_ * 0.9)
                            shl = 3.7 * li_ + (0.0 if inb else 5.0)
                            ncell = int(math.floor((zg + shl) / 9.5))
                            car = _pulse(zg + shl, fz, 9.5, 2.0, 4.4) * acr
                            if _hash(ncell, li_, 5 if inb else 6, 501) > 0.25 + 0.55 * den:
                                car = 0.0
                            if inb:
                                er, eg, eb = 1.7 * car, 1.4 * car, 0.95 * car
                            else:
                                er, eg, eb = 1.5 * car, 0.14 * car, 0.07 * car
                            # lane markings
                            mk = _pulse(zg, fz, 12.0, 0.0, 6.0) * _sstep(0.12 + fp, 0.02, abs(lane - li_))
                            cr += 0.14 * mk
                            cg += 0.13 * mk
                            cb += 0.15 * mk
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
                ytop = hb
                pk = 0
                rk = int(B[i, RK])
                if rk > 0 and mat != M_FENCE:
                    # round 11: pitched / slanted roofs (gable houses, street-slant setbacks): the nearest valid
                    # hit among the front face, the visible side face and the roof planes
                    rs = B[i, RS]
                    zbest = 1e18
                    # front face
                    X = dx * z0 / f
                    Y = cam_h - dy * z0 / f
                    if X >= x0 and X <= x1 and Y >= base:
                        tp = _roof_top(rk, rs, x0, x1, z0, z1, hb, base, X, z0)
                        if Y <= tp:
                            zbest = z0
                            face = 0
                            u = X - x0
                            v = Y - base
                            fu = z0 / f / ss
                            fv = fu
                            Yh = Y
                            ytop = tp
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
                        if zs >= z0 and zs <= z1 and Y >= base and zs < zbest:
                            tp = _roof_top(rk, rs, x0, x1, z0, z1, hb, base, xs, zs)
                            if Y <= tp:
                                zbest = zs
                                face = sd
                                u = zs - z0
                                v = Y - base
                                fv = zs / f / ss
                                fu = fv * zs / abs(xs) if abs(xs) > 1e-3 else 1e3
                                Yh = Y
                                ytop = tp
                    # roof planes y = a + bx * x + bz * z
                    for pi_ in range(3):
                        a_ = 0.0
                        bx_ = 0.0
                        bz_ = 0.0
                        okp = False
                        if rk == 1:
                            dsl = _slant_depth(rs, z0, z1, hb, base)
                            if pi_ == 0:
                                a_ = hb - rs * (z0 + dsl)
                                bz_ = rs
                                okp = True
                            elif pi_ == 1:
                                a_ = hb
                                okp = True
                        elif rk == 2:
                            zm = 0.5 * (z0 + z1)
                            if pi_ == 0:
                                a_ = hb - rs * zm
                                bz_ = rs
                                okp = True
                            elif pi_ == 1:
                                a_ = hb + rs * zm
                                bz_ = -rs
                                okp = True
                        else:
                            xm = 0.5 * (x0 + x1)
                            if pi_ == 0:
                                a_ = hb - rs * xm
                                bx_ = rs
                                okp = True
                            elif pi_ == 1:
                                a_ = hb + rs * xm
                                bx_ = -rs
                                okp = True
                        if not okp:
                            continue
                        den = dy / f + bx_ * dx / f + bz_
                        if den <= 1e-9:
                            continue
                        zr = (cam_h - a_) / den
                        if zr < z0 or zr > z1 or zr >= zbest:
                            continue
                        X = dx * zr / f
                        if X < x0 or X > x1:
                            continue
                        Y = cam_h - dy * zr / f
                        tp = _roof_top(rk, rs, x0, x1, z0, z1, hb, base, X, zr)
                        if abs(Y - tp) > 0.02 + 0.002 * zr / f * 4.0 or Y < base:
                            continue
                        zbest = zr
                        face = 4
                        if rk == 1 and pi_ == 1:
                            face = 3
                        pk = pi_
                        u = X - x0
                        v = zr - z0
                        Yh = Y
                        ytop = tp
                    if face >= 0:
                        zz = zbest
                        Y = Yh
                if rk > 0 and mat != M_FENCE:
                    pass
                else:
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
                if face == 3 and mat == M_ROAD:
                    # round 12: expressway deck seen from above: asphalt, parapets, dashed lane lines, a
                    # median, lamp posts, and traffic as small light dots (white-gold inbound / red outbound)
                    wa = x1 - x0
                    ua = u
                    zw_ = z0 + v
                    fz = zz / f / ss * zz / max(cam_h - hb, 1.0)
                    pwr = zz / f / ss
                    cr, cg, cb = 0.09, 0.085, 0.12
                    er, eg, eb, eon, ew = 0.0, 0.0, 0.0, -10.0, 0.0
                    if ua < 0.7 + pwr or ua > wa - 0.7 - pwr:
                        cr, cg, cb = 0.34, 0.31, 0.4
                        lp = _pulse(zw_, fz, 36.0, 0.0, 1.6)
                        er, eg, eb, ew = 1.3 * lp, 0.8 * lp, 0.36 * lp, lp
                    elif abs(ua - 0.5 * wa) < 0.5 + pwr * 0.5:
                        cr, cg, cb = 0.2, 0.19, 0.24
                    else:
                        inb = ua < 0.5 * wa
                        lane = (ua - 0.7) / 3.4 if inb else (ua - 0.5 * wa - 0.5) / 3.4
                        lnr_ = int(lane)
                        dl = abs(lane - lnr_ - 0.5)
                        acr = _sstep(0.3 + pwr * 0.3, 0.15, dl)
                        shl = 5.3 * lnr_ + (0.0 if inb else 7.0)
                        ncell = int(math.floor((zw_ + shl) / 11.0))
                        car = _pulse(zw_ + shl, fz, 11.0, 2.0, 4.4) * acr
                        den = 0.5 + 0.5 * math.sin(zw_ * 0.006 + lnr_ * 1.3 + (0.0 if inb else 2.0))
                        if _hash(ncell, lnr_, 7 if inb else 8, 502) > 0.18 + 0.42 * den:
                            car = 0.0
                        if inb:
                            er, eg, eb = 1.6 * car, 1.35 * car, 0.95 * car
                        else:
                            er, eg, eb = 1.15 * car, 0.22 * car, 0.12 * car
                        ew = car
                        mk = _pulse(zw_, fz, 10.0, 0.0, 5.0) * _sstep(0.12 + pwr, 0.03, abs(lane - lnr_))
                        cr += 0.16 * mk
                        cg += 0.15 * mk
                        cb += 0.16 * mk
                        cr += 0.04 * car
                        cg += 0.03 * car
                elif face == 3:
                    # roof: reflects the violet zenith; material by roof type; thin parapet rim, the far
                    # (sunward) parapet catches a pink glint
                    ar, ag, ab = B[i, AR], B[i, AG], B[i, AB]
                    kf = v / max(z1 - z0, 1.0)
                    pw = zz / f / ss * 1.2
                    edge = _sstep(pw * 1.2 + 0.25, 0.0, (z1 - z0) - v)
                    edge2 = max(_sstep(pw * 1.5 + 0.3, 0.0, v), _sstep(pw * 1.5 + 0.3, 0.0, u),
                                _sstep(pw * 1.5 + 0.3, 0.0, (x1 - x0) - u))
                    rt = int(B[i, ROOFT])
                    sdr = int(B[i, SEED])
                    rv = _hash(sdr, 7, 7, 7)
                    # round 13: painted roof planes. Flat roofs look straight up at the bright dusk sky, so
                    # they are the lightest planes of the rooftop sea: lit by a lilac sky dome, warming toward
                    # the afterglow axis. Real roof materials (grey concrete, beige tile, green / grey membrane,
                    # dark tar, rust-red painted, white panel), membrane seams, drain stains, weathering.
                    mt = (rt + int(rv * 5.0)) % 6
                    if mt == 0:     # grey concrete
                        rr_, rg_, rb_ = 0.56, 0.55, 0.58
                    elif mt == 1:   # beige tile / sand
                        rr_, rg_, rb_ = 0.66, 0.56, 0.45
                    elif mt == 2:   # green membrane
                        rr_, rg_, rb_ = 0.36, 0.5, 0.44
                    elif mt == 3:   # dark tar / asphalt sheet
                        rr_, rg_, rb_ = 0.27, 0.26, 0.31
                    elif mt == 4:   # rust-red painted metal / brick tile
                        rr_, rg_, rb_ = 0.58, 0.34, 0.28
                    else:           # white panel / light membrane
                        rr_, rg_, rb_ = 0.76, 0.76, 0.78
                    kv_ = 0.8 + 0.4 * _hash(sdr, 17, 3, 5)
                    axr = math.exp(-abs((x0 + x1) * 0.5 / zz - sun_xw) * 5.0)
                    lr_ = 0.74 + 0.4 * axr
                    lg_ = 0.7 + 0.16 * axr
                    lb_ = 0.8 - 0.14 * axr
                    cr = rr_ * lr_ * kv_ * (0.7 + 0.3 * ar) * 1.05
                    cg = rg_ * lg_ * kv_ * (0.7 + 0.3 * ag) * 1.05
                    cb = rb_ * lb_ * kv_ * (0.7 + 0.3 * ab) * 1.05
                    # grazing light: the far part of the roof (toward the afterglow) is brighter, the near part
                    # darker (sky reflection falloff)
                    gz = 0.85 + 0.25 * kf
                    cr, cg, cb = cr * gz, cg * gz, cb * gz
                    # membrane seams / tile courses (parallel to x) and a few cross joints
                    pwm = zz / f / ss
                    sm = _pulse(v, pwm * zz / max(cam_h - hb, 1.0), 3.2 + 1.6 * rv, 0.0, 0.18) * 0.5 +                          _pulse(u, pwm, 6.0 + 3.0 * rv, 1.0, 0.2) * 0.3
                    # weathering: soft stain blotches + drain streaks from the corners
                    stn = 0.5 + 0.5 * math.sin(u * 0.61 + sdr * 0.37) * math.sin(v * 0.83 + sdr * 0.11)
                    stn = _sstep(0.55, 0.95, stn) * 0.18
                    drn = _sstep(2.5, 0.0, min(u, (x1 - x0) - u)) * 0.15
                    wk = 1.0 - sm * 0.14 - stn - drn
                    cr, cg, cb = cr * wk, cg * wk, cb * (wk * 0.97 + 0.03)
                    # parapet lines; the far (sunward) parapet catches a warm glint near the sun axis
                    ek = (0.2 + 0.8 * axr * axr) * (1.0 if _hash(sdr, 71, 73, 79) < 0.6 else 0.35)
                    cr = cr * (1 - 0.45 * edge2) + edge * 0.75 * ek
                    cg = cg * (1 - 0.45 * edge2) + edge * 0.42 * ek
                    cb = cb * (1 - 0.4 * edge2) + edge * 0.24 * ek
                    # helipad on a few large roofs
                    er, eg, eb, eon, ew = 0.0, 0.0, 0.0, 0.0, 0.0
                    # round 11: the low sun slips between the towers and catches whole districts of roofs
                    xw_ = x0 + u
                    zw_ = z0 + v
                    spt = _sstep(0.58, 0.8, 0.5 + 0.32 * math.sin(xw_ * 0.011 + zw_ * 0.0042 + 0.7) +
                                 0.22 * math.sin(xw_ * 0.029 - zw_ * 0.009 + 1.3))
                    cr += spt * 0.42 * (0.6 + 0.4 * rv)
                    cg += spt * 0.24 * (0.6 + 0.4 * rv)
                    cb += spt * 0.1 * (0.6 + 0.4 * rv)
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
                elif face == 4:
                    # round 11: pitched roof plane (kawara tile / painted metal / slate), lit by orientation:
                    # the back slope and the -x slope face the afterglow (warm grazing light), the front slope
                    # faces the camera and the eastern sky (cool, lighter than the walls), the +x slope is shade
                    rt = int(B[i, ROOFT])
                    sdr = int(B[i, SEED])
                    pwr = zz / f / ss
                    if rt == 0:
                        br0, bg0, bb0 = 0.26, 0.28, 0.36      # dark kawara blue-grey
                    elif rt == 1:
                        br0, bg0, bb0 = 0.5, 0.24, 0.19       # red-brown painted metal
                    elif rt == 2:
                        br0, bg0, bb0 = 0.2, 0.38, 0.38       # teal / verdigris metal
                    elif rt == 3:
                        br0, bg0, bb0 = 0.22, 0.25, 0.42      # navy metal
                    else:
                        br0, bg0, bb0 = 0.55, 0.54, 0.56      # silver slate / galvanised
                    if (rk == 3 and pk == 0) or (rk != 3 and pk == 1):
                        lr_, lg_, lb_ = 1.35, 0.82, 0.5
                    elif rk == 3:
                        lr_, lg_, lb_ = 0.3, 0.32, 0.52
                    else:
                        lr_, lg_, lb_ = 0.62, 0.66, 0.92
                    jv = 0.85 + 0.3 * _hash(sdr, 3, 9, 27)
                    cr, cg, cb = br0 * lr_ * jv, bg0 * lg_ * jv, bb0 * lb_ * jv
                    # tile courses parallel to the eaves
                    wcrd = u if rk == 3 else v
                    crs = _pulse(wcrd, pwr * 1.5, 0.45, 0.0, 0.1)
                    cr, cg, cb = cr * (1 - 0.18 * crs), cg * (1 - 0.18 * crs), cb * (1 - 0.15 * crs)
                    # ridge line: thin highlight where the slopes meet (lit side) / dark on the shade side
                    if rk >= 2:
                        dr_ = abs(u - 0.5 * (x1 - x0)) if rk == 3 else abs(v - 0.5 * (z1 - z0))
                        rg_ = _sstep(pwr * 2.0 + 0.2, 0.0, dr_)
                        cr, cg, cb = cr + 0.3 * rg_, cg + 0.2 * rg_, cb + 0.16 * rg_
                    er, eg, eb, eon, ew = 0.0, 0.0, 0.0, 0.0, 0.0
                    if rk == 1:
                        # round 13: a street-slant setback is not a blank slope - it is a stack of stepped-back
                        # floors: sky-lit terrace strips, window rows in the building's own material, the odd
                        # lit office, a warm line on the terrace edges near the sun axis
                        flh_s = B[i, FLH]
                        fr_s = (Yh / flh_s) - math.floor(Yh / flh_s)
                        fl_s = int(math.floor(Yh / flh_s))
                        am_ = (B[i, AR] + B[i, AG] + B[i, AB]) / 3.0
                        a0_ = am_ + (B[i, AR] - am_) * 1.6
                        a1_ = am_ + (B[i, AG] - am_) * 1.6
                        a2_ = am_ + (B[i, AB] - am_) * 1.6
                        wr_s, wg_s, wb_s = 0.27 * a0_, 0.26 * a1_, 0.28 * a2_
                        terr = _sstep(0.7, 0.8, fr_s)
                        wrow = _sstep(0.12, 0.18, fr_s) * (1.0 - _sstep(0.58, 0.64, fr_s))
                        bw_s = B[i, BAYW]
                        wu_s = _pulse(u - 0.2 * bw_s, pwr, bw_s, 0.0, 0.65 * bw_s)
                        wmk = wrow * wu_s
                        hwn = _hash(sdr, fl_s, int(math.floor(u / (bw_s * 2.0))), 617)
                        if hwn < 0.35:
                            gr_s, gg_s, gb_s = 0.3, 0.28, 0.26
                        else:
                            gr_s, gg_s, gb_s = 0.17, 0.21, 0.29
                        axs = math.exp(-abs((x0 + x1) * 0.5 / zz - sun_xw) * 5.0)
                        tr_s = 0.5 + 0.25 * axs
                        cr = (wr_s * (1 - wmk) + gr_s * wmk) * (1 - terr) + tr_s * 1.05 * terr
                        cg = (wg_s * (1 - wmk) + gg_s * wmk) * (1 - terr) + tr_s * 0.95 * terr
                        cb = (wb_s * (1 - wmk) + gb_s * wmk) * (1 - terr) + tr_s * (1.02 - 0.2 * axs) * terr
                        eg_ = _sstep(0.9, 0.97, fr_s) * (0.15 + 0.6 * axs * axs)
                        cr, cg, cb = cr + 0.9 * eg_, cg + 0.5 * eg_, cb + 0.28 * eg_
                        if hwn > 0.86 and wmk > 0.0:
                            er, eg, eb = 0.9 * wmk, 0.75 * wmk, 0.5 * wmk
                            eon = 0.3 + 4.6 * _hash(sdr, fl_s, 3, 619)
                            ew = wmk
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
                        simp = 0.6 * _sstep(5.0, 2.2, ppf)
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
                                # round 13: sky-coloured glass - dusty apricot low, lilac high, per-block value
                                vv_ = 0.8 + 0.4 * _hash(sd_, 13, 17, 19)
                                tr_ = (0.36 + 0.12 * hv_) * vv_
                                tg_ = (0.27 + 0.1 * hv_) * vv_
                                tb_ = (0.3 + 0.24 * hv_) * vv_
                            tr_ += 0.14 * dg_
                            tg_ += 0.07 * dg_
                            tb_ += 0.08 * dg_
                            tmp[0] = tmp[0] * (1.0 - gk_) + tr_ * gk_
                            tmp[1] = tmp[1] * (1.0 - gk_) + tg_ * gk_
                            tmp[2] = tmp[2] * (1.0 - gk_) + tb_ * gk_
                    cr, cg, cb = tmp[0], tmp[1], tmp[2]
                    if rk > 0:
                        # eave overhang shadow on the wall just under a pitched roof
                        ea_ = 0.5 + 0.5 * _sstep(0.0, 1.2 + zz / f, ytop - Y)
                        cr, cg, cb = cr * ea_, cg * ea_, cb * ea_
                    if hb - base <= 45.0 and face != 3 and mat != M_BILL and mat != M_NAME:
                        # round 9: painted value breaks between neighbouring mid-rise facades (each block its
                        # own value step) and a lighter top fading to a darker base (one broad gradient)
                        sdv = int(B[i, SEED])
                        vb_ = 0.68 + 0.64 * _hash(sdv, 29, 31, 37)
                        ht_ = _hash(sdv, 41, 43, 47) - 0.5          # warm / cool paint variation
                        hv2 = (Y - base) / max(hb - base, 1.0)
                        vg_ = 0.82 + 0.3 * hv2
                        cr = cr * vb_ * vg_ * (1.0 + 0.45 * ht_)
                        cg = cg * vb_ * vg_
                        cb = cb * vb_ * vg_ * (1.0 - 0.35 * ht_)
                    # round 11: rims only where the light can reach. The low sun sits behind the city, so the
                    # camera-facing walls are in shade; only the silhouette edges and roof lines toward the sun
                    # catch a warm line, and only on some blocks (towers paint their own edges in _tower)
                    er, eg, eb = tmp[3], tmp[4], tmp[5]
                    eon = tmp[6]
                    ew = tmp[7]
                    pw = zz / f / ss
                    tl = _sstep(pw * 2.0 + 0.4, 0.0, ytop - Y)
                    sdr_ = int(B[i, SEED])
                    ax = math.exp(-abs((x0 + x1) * 0.5 / zz - sun_xw) * 6.0)
                    lucky = _hash(sdr_, 71, 73, 79)
                    if not twr and face == 0 and mat != M_BILL and mat != M_NAME:
                        wdt = max(x1 - x0, 1.0)
                        side = 1.0 if (x0 + x1) * 0.5 > sun_xw * zz else -1.0
                        pwr = zz / f
                        dm = u if side > 0 else wdt - u
                        rl = _sstep(pwr * 1.7, pwr * 0.5, dm) * (0.5 * ax * ax + 0.12) * (1.0 if lucky < 0.5 else 0.45)
                        # round 14: the side away from the sun is a lost edge that melts into the haze
                        dsh = wdt - dm
                        lost = _sstep(pwr * 6.0, 0.0, dsh) * 0.4
                        cr = cr * (1 - lost) + 0.3 * lost
                        cg = cg * (1 - lost) + 0.29 * lost
                        cb = cb * (1 - lost) + 0.42 * lost
                        rt_ = _sstep(pwr * 1.4 + 0.1, 0.0, ytop - Y) * (0.45 * ax * ax + 0.08) * (1.0 if lucky < 0.35 else 0.0)
                        rr_ = max(rl, rt_)
                        cr += rr_ * 1.3
                        cg += rr_ * 0.72
                        cb += rr_ * 0.42
                        # the parapet cap: a slightly lighter cool line on some roofs only
                        # round 13: every roof line catches the open sky (the rooftop sea reads as stacked light
                        # parapet lines, yn_07 / yn_08); warmer toward the afterglow
                        kcap = 0.26 if lucky > 0.6 else 0.14
                        cr += tl * kcap * (1.0 + 0.5 * ax)
                        cg += tl * kcap * (0.95 + 0.2 * ax)
                        cb += tl * kcap * (1.15 - 0.2 * ax)
                    elif not twr and face == 1:
                        # the sunward side wall: grazing warm light, top edge + far corner catch it
                        cr += tl * (0.2 + 0.3 * lucky)
                        cg += tl * (0.11 + 0.17 * lucky)
                        cb += tl * 0.08
                        ce = _sstep(pw * 2.5 * zz / max(abs(x0), 1.0) + 0.6, 0.0, z1 - zz)
                        gl = 0.8 if (mat == M_GLASS or mat == M_DARK) else 0.35
                        cr += ce * 1.1 * gl
                        cg += ce * 0.7 * gl
                        cb += ce * 0.4 * gl
                        if (mat == M_GLASS or mat == M_DARK or mat == M_CURTAIN or mat == M_FINS) and hb > 45.0:
                            dpth = max(z1 - z0, 1.0)
                            q = (zz - z0) / dpth
                            hv = (Y - base) / max(hb - base, 1.0)
                            dg = q * 0.8 - hv + 0.55 + 0.25 * _hash(int(B[i, SEED]), 3, 1, 9)
                            band = math.exp(-(dg / 0.13) ** 2) * _sstep(0.2, 0.75, hv)
                            gk = band * (0.6 if mat != M_FINS else 0.3)
                            if gk > 0.01:
                                cr += gk * 0.9
                                cg += gk * 0.5
                                cb += gk * 0.25
                                er += gk * 1.0
                                eg += gk * 0.56
                                eb += gk * 0.25
                                if eon > -5.0:
                                    eon = -10.0
                                ew = max(ew, gk)
                    elif not twr and face == 2:
                        cr += tl * 0.03
                        cg += tl * 0.03
                        cb += tl * 0.05
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
            cm = 0.4 * (1.0 - ks)
            yw = max(yw_hit, 0.0)
            dd = max(zh - 700.0, 0.0) / fogZ
            # the veil gets lighter / bluer with distance (aerial perspective)
            dk = min(max((zh - 2000.0) / 5000.0, 0.0), 1.0)
            # round 6: the veil is a pink-lavender dusk haze that pales with distance (aerial perspective),
            # thicker low down between the mid-rise blocks
            hr = hr * (1 - cm) + (0.24 + 0.14 * dk) * cm
            hg = hg * (1 - cm) + (0.27 + 0.15 * dk) * cm
            hb2 = hb2 * (1 - cm) + (0.38 + 0.16 * dk) * cm
            # round 13: the near / mid blocks keep their material colours (the depth haze is laid per plate)
            fa = 1.0 - math.exp(-(0.1 * dd + 0.55 * dd ** 1.6 + 0.3 * dd * math.exp(-yw / 65.0)))
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


@njit(cache=True, parallel=True)
def city_grade(img, cm, gw, gx0, kb):
    """round 11: screen-space warm / cool split of the city (see Scene._city_grade), in place.
    gw: padded sun-cone weight (H, Wp); gx0: its x offset this frame; kb: (H,) indigo base weight."""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        kby = kb[y]
        for x in range(W):
            c = cm[y, x]
            if c <= 1e-4:
                continue
            wc = gw[y, x + gx0]
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            lum = 0.3 * r + 0.55 * g + 0.15 * b
            # the gold glass hits / sun-lit edges keep their colour (selective warm accents)
            prot = min(max((r - b - 0.12) / 0.3, 0.0), 1.0) * min(max((lum - 0.22) / 0.25, 0.0), 1.0)
            c = c * (1.0 - 0.85 * prot)
            sk = 1.0 - 0.1 * (1.0 - wc)
            r, g, b = lum + (r - lum) * sk, lum + (g - lum) * sk, lum + (b - lum) * sk
            # round 13: the cool side is only a little deeper / bluer (it was one lilac-blue wash that ate
            # the material colours)
            # round 14: shade away from the sun is a saturated blue-violet (real warm / cool split);
            # only the dark shadow values are pushed, the lit accents stay
            dk = 0.86 + 0.14 * wc
            shd = (1.0 - wc) * (1.0 - min(max((lum - 0.25) / 0.35, 0.0), 1.0))
            r = r * (0.88 + 0.12 * wc - 0.08 * shd) * dk
            g = g * (0.93 + 0.07 * wc - 0.1 * shd) * dk
            b = b * (1.05 - 0.05 * wc + 0.14 * shd) * dk
            pk = 0.3 * wc * (0.35 + 0.65 * min(lum * 2.5, 1.0))
            r = r * (1.0 - 0.3 * wc) + 1.0 * pk
            g = g * (1.0 - 0.3 * wc) + 0.66 * pk
            b = b * (1.0 - 0.3 * wc) + 0.42 * pk
            r = r * (1.0 - kby * 0.5)
            g = g * (1.0 - kby * 0.48)
            b = b * (1.0 - kby * 0.2)
            img[y, x, 0] = img[y, x, 0] * (1.0 - c) + r * c
            img[y, x, 1] = img[y, x, 1] * (1.0 - c) + g * c
            img[y, x, 2] = img[y, x, 2] * (1.0 - c) + b * c


@njit(cache=True, parallel=True)
def sun_compress(img, scw, gx0, keep, use_keep):
    """per-channel highlight shoulder around the sun (see Scene._sun_compress), in place"""
    H, W = img.shape[0], img.shape[1]
    k0, top = 0.75, 0.42
    for y in prange(H):
        for x in range(W):
            wm = scw[y, x + gx0]
            if use_keep:
                wm = wm * (1.0 - 0.85 * min(max(keep[y, x], 0.0), 1.0))
            if wm < 1e-4:
                continue
            for c in range(3):
                v = img[y, x, c]
                d = v - k0
                if d > 0.0:
                    f = v - d + d / (1.0 + d / top)
                    img[y, x, c] = v * (1.0 - wm) + f * wm


@njit(cache=True, parallel=True)
def add_layer(img, lay, gx0, cl, k):
    """img += lay[:, gx0:gx0+W] * (1 - k * cl)"""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            m = 1.0 - k * cl[y, x]
            for c in range(3):
                img[y, x, c] += lay[y, x + gx0, c] * m


# ---------------------------------------------------------------------------------------------------------------
# round 13: fused per-frame post kernels (frame budget)
@njit(cache=True, fastmath=True, parallel=True)
def post_add(img, em, gup, sh, ca, fga, glare, gx0):
    """img += em + gup + (sh * (1 - .75 ca) + glare * .45 * (1 - .6 ca)) * (1 - .7 fga), in place"""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            c_ = ca[y, x]
            nk = 1.0 - 0.7 * min(max(fga[y, x], 0.0), 1.0)
            k1 = (1.0 - 0.75 * c_) * nk
            k2 = 0.45 * (1.0 - 0.6 * c_) * nk
            for c in range(3):
                img[y, x, c] += em[y, x, c] + gup[y, x, c] + sh[y, x, c] * k1 + glare[y, x + gx0, c] * k2


@njit(cache=True, fastmath=True)
def _shl(v, s, k):
    if v > s:
        o = v - s
        return s + k * o / (o + k)
    return v


@njit(cache=True, fastmath=True, parallel=True)
def shoulder2(img, hv, s1, d1, s2, d2):
    """F.shoulder(img, s1, d1) * (1 - hv) + F.shoulder(img, s2, d2) * hv in one pass (in place)"""
    H, W = img.shape[0], img.shape[1]
    k1, k2 = 1.0 - s1, 1.0 - s2
    for i in prange(H):
        for j in range(W):
            h = hv[i, j]
            m = 0.0
            for c in range(3):
                v = max(img[i, j, c], 0.0)
                if v > m:
                    m = v
            wd1 = min(max((m - 1.0) / 1.5, 0.0), 1.0) * d1
            wd2 = min(max((m - 1.0) / 1.5, 0.0), 1.0) * d2
            a0 = _shl(max(img[i, j, 0], 0.0), s1, k1)
            a1 = _shl(max(img[i, j, 1], 0.0), s1, k1)
            a2 = _shl(max(img[i, j, 2], 0.0), s1, k1)
            b0 = _shl(max(img[i, j, 0], 0.0), s2, k2)
            b1 = _shl(max(img[i, j, 1], 0.0), s2, k2)
            b2 = _shl(max(img[i, j, 2], 0.0), s2, k2)
            ya = max(a0, max(a1, a2))
            yb = max(b0, max(b1, b2))
            a0, a1, a2 = a0 * (1 - wd1) + ya * wd1, a1 * (1 - wd1) + ya * wd1, a2 * (1 - wd1) + ya * wd1
            b0, b1, b2 = b0 * (1 - wd2) + yb * wd2, b1 * (1 - wd2) + yb * wd2, b2 * (1 - wd2) + yb * wd2
            img[i, j, 0] = a0 * (1 - h) + b0 * h
            img[i, j, 1] = a1 * (1 - h) + b1 * h
            img[i, j, 2] = a2 * (1 - h) + b2 * h


@njit(cache=True, fastmath=True, parallel=True)
def paper_bloom(img, paper, occ, occ_sky, base, gup, gk):
    """paper surface (mid-tone weighted multiply, weight base..1 by the painted-city coverage) + a gentle
    overall bloom (gup upsampled glow * gk), in place"""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            cm = min(max(occ[y, x] - occ_sky[y, x], 0.0), 1.0)
            tex = (paper[y, x] - 1.0) * (base + (1.0 - base) * cm)
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            lum = 0.3 * r + 0.55 * g + 0.15 * b
            w = min(max(4.0 * lum * (1.2 - lum), 0.25), 1.0)
            f = 1.0 + tex * w
            img[y, x, 0] = r * f + gup[y, x, 0] * gk
            img[y, x, 1] = g * f + gup[y, x, 1] * gk
            img[y, x, 2] = b * f + gup[y, x, 2] * gk
