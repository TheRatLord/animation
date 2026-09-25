"""s10 helper - the painted cloud FLOOR (near sea of clouds, below the summit lip).

Laid out in perspective: rows of cumulus heads whose spacing and lobe size grow toward the camera (near
heads 2-3x the plain perspective size). Each HEAD is one painted mass whose silhouette is the union of
lobes on lobes (dome -> crown lobes -> bumps -> fine bumps on the big near heads); lower / nearer
sub-heads (skirts) are painted after it, so their own crisp tops lay over the cool core of the head
behind (the Shinkai stacking). Heads are painted by a numba kernel as flat value masses, not spheres:
  * a cool, saturated blue-violet core (aerial perspective: hazy lilac far, deep violet-blue near);
  * a lighter / warmer 'scatter' mass on the upper, light-facing part of the head (warm light inside the
    heads near the sun column, cool lilac sky light away from it);
  * a lit crust hugging the upper light-facing contour, its width breaking along the edge, with a hard
    brushed inner border and faint cusp notches between the cauliflower bumps;
  * the down-sun flank darker with a soft LOST silhouette; the base dissolves into the bank below;
  * a thin hot lining only on exposed sun-facing crowns of lit heads, strongest on the sun axis and fading
    to nothing toward the frame edges (separate additive plate so the light can swell it).
Below every row its bank continues down as a dark slab, so the gaps between rows fall into deep indigo.
"""
import math
import numpy as np
from numba import njit, prange

F32 = np.float32
CAUL = True
FACE_K = 0.42
UPW = 0.6
WARM_K = 0.55
GROW = (0.85, 1.9, 1.2)


def _noise(w, h, cells, seed, octaves=3, stretch=1.0):
    from lib import clouds3 as K
    return K._noise(w, h, cells, seed, octaves, stretch, 0.0).astype(F32)


def ss_py(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a, b, t):
    return np.asarray(a, F32) * (1 - t) + np.asarray(b, F32) * t


# palettes (far -> near)
# far rows: hazy, low contrast (core / mid / shade close together); near rows: deep cool blue-violet
FAR = dict(core=(0.36, 0.33, 0.56), mid=(0.62, 0.52, 0.7), shade=(0.24, 0.24, 0.47), slab=(0.26, 0.26, 0.47),
           valley=(0.16, 0.17, 0.36), lit_cool=(0.98, 0.74, 0.74), lit_hot=(1.3, 0.86, 0.56))
# near rows: opaque volumes - lit top -> warm mauve mid -> violet base (no near-black cavities)
NEAR = dict(core=(0.17, 0.145, 0.36), mid=(0.46, 0.34, 0.54), shade=(0.06, 0.05, 0.2), slab=(0.07, 0.06, 0.21),
            valley=(0.035, 0.03, 0.13), lit_cool=(0.86, 0.6, 0.66), lit_hot=(1.3, 0.72, 0.4))
MAUVE = (0.52, 0.34, 0.46)            # warm mauve mid-value between the lit tops and the violet base
SCAT_HOT = (0.74, 0.46, 0.4)          # warm light scattered inside the sun-side heads
ROSE = (0.7, 0.44, 0.58)              # rose light on the sun-facing flanks right of the sun column
RIM_HOT = (2.1, 1.2, 0.55)

# head record layout
HF = 44
(H_X0, H_Y0, H_X1, H_Y1, H_LX, H_LY, H_CORE, H_MID, H_LIT, H_SHD, H_RIM, H_LAMT, H_MAMT, H_RAMT, H_LOST, H_CW,
 H_KIND, H_BF, H_NOFF, H_HX, H_HT, H_HH, H_HW, H_I0, H_I1, H_DN, H_SLO, H_SLX, H_SLY) = (
    0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38)
H_MV, H_OPQ, H_MVC = 39, 40, 41     # mauve amount, base opacity, mauve colour (3)


@njit(cache=True, fastmath=True)
def _ss(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def _cauli(dx, dy, r, m, li, n2, sc):
    """Cauliflower scallops on the UPPER contour of a lobe: rounded bumps separated by sharp inward
    notches (|sin| of the lobe angle), sized ~6-11 px, amplitude grows with the lobe; none underneath."""
    if m < 3.0 * sc:
        return 0.0
    upf = -dy / (r + 1e-6)
    if upf < -0.05:
        return 0.0
    hsh = (li * 0.6180339) % 1.0
    if hsh < 0.3:
        return 0.0                   # some lobes stay smooth (no zipper of identical bumps)
    amp = min(0.05 * m, 1.7 * sc) * _ss(-0.05, 0.5, upf) * _ss(-0.6, 0.3, n2)
    bs = (4.5 + 5.0 * hsh + 3.0 * _ss(8.0, 60.0, m / sc)) * sc
    nb = max(math.floor(2.0 * math.pi * m / bs), 3.0)
    th = math.atan2(dy, dx)
    return amp * (abs(math.sin(0.5 * nb * th + 1.7 * li + 2.2 * n2)) - 0.62)


@njit(cache=True, fastmath=True)
def _paint(C, A, R, Hd, Lb, N1, N2, N3, sc):
    """Hd: head records; Lb: lobes (cx, cy, rx, ry, wob, level) referenced by [H_I0, H_I1)."""
    PH, PW = A.shape
    for hi in range(Hd.shape[0]):
        kind = Hd[hi, H_KIND]
        if kind == 1.0:
            # ---- row slab: from a wobbling base line down, body -> valley gradient
            yb, r0 = Hd[hi, H_HT], Hd[hi, H_HH]
            fe = max(Hd[hi, H_LOST], 1.0)          # feathered slab top (far rows: a hazy lost edge)
            for x in range(PW):
                base = yb + r0 * 0.35 * N3[min(max(int(yb), 0), PH - 1), x]
                y0 = int(max(base - fe - 2, 0))
                y1 = int(min(yb + 6.0 * r0 + 4, PH))
                for y in range(y0, y1):
                    a = (y - base + 0.5 * fe) / fe
                    if a <= 0.0:
                        continue
                    if a > 1.0:
                        a = 1.0
                    k = _ss(0.0, 2.0 * r0, y - base + 0.4 * r0 * N2[y, x])
                    for c in range(3):
                        v = Hd[hi, H_CORE + c] * (1 - k) + Hd[hi, H_SHD + c] * k
                        C[y, x, c] = C[y, x, c] * (1 - a) + v * a
                        R[y, x, c] = R[y, x, c] * (1 - a)
                    A[y, x] = A[y, x] + a * (1 - A[y, x])
            continue
        _paint_head(C, A, R, Hd, hi, Lb, N1, N2, N3, sc)


@njit(cache=True, fastmath=True, parallel=True)
def _paint_head(C, A, R, Hd, hi, Lb, N1, N2, N3, sc):
    PH, PW = A.shape
    i0, i1 = int(Hd[hi, H_I0]), int(Hd[hi, H_I1])
    x0 = int(max(Hd[hi, H_X0], 0))
    x1 = int(min(Hd[hi, H_X1], PW))
    y0 = int(max(Hd[hi, H_Y0], 0))
    y1 = int(min(Hd[hi, H_Y1], PH))
    if x1 <= x0 + 2 or y1 <= y0:
        return
    lx, ly = Hd[hi, H_LX], Hd[hi, H_LY]
    lamt, mamt, ramt = Hd[hi, H_LAMT], Hd[hi, H_MAMT], Hd[hi, H_RAMT]
    lost, cwp, bf, dn = Hd[hi, H_LOST], Hd[hi, H_CW], Hd[hi, H_BF], Hd[hi, H_DN]
    hx, ht, hh, hw = Hd[hi, H_HX], Hd[hi, H_HT], Hd[hi, H_HH], Hd[hi, H_HW]
    noff = int(Hd[hi, H_NOFF])
    slo = Hd[hi, H_SLO]
    slx, sly = Hd[hi, H_SLX], Hd[hi, H_SLY]
    mauv = Hd[hi, H_MV]
    opq = Hd[hi, H_OPQ]
    crisp = 0.55 + 0.25 * sc
    # ---- pass 1: the crest line (first inside pixel of each column, sub-pixel) -> the painted light
    # follows the TOP silhouette only: lit crust / rim on crest segments facing the sun, tapering to
    # nothing where the crest turns steep or away; interiors and undersides never get an outline
    nwc = x1 - x0
    top = np.full(nwc, 1e9, np.float32)
    for x in prange(x0, x1):
        xn = x + noff
        if xn >= PW:
            xn = 2 * PW - 1 - xn
        ep = -1e9
        for y in range(y0, y1):
            n1 = N1[y, xn]
            n2 = N2[y, xn]
            e1 = -1e9
            for li in range(i0, i1):
                cx, cy, rx, ry, wob = Lb[li, 0], Lb[li, 1], Lb[li, 2], Lb[li, 3], Lb[li, 4]
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                r = math.sqrt(dx * dx + dy * dy)
                m = min(rx, ry)
                if (r - 1.0) * m > 4.0 + wob * m:
                    continue
                e = (1.0 - r) * m - r * min(wob * m, 1.2 + 1.2 * sc) * (0.6 * n1 + 0.4 * n2)
                if CAUL and abs(e) < 4.0 * sc + 3.0:
                    e += _cauli(dx, dy, r, m, li, n2, sc)
                if e > e1:
                    e1 = e
            if e1 >= 0.0:
                fr = 0.0
                if ep > -1e8 and e1 - ep > 1e-4:
                    fr = e1 / (e1 - ep)
                top[x - x0] = y - fr
                break
            ep = e1
    # smoothed crest + its slope (lobe-scale normals: each cauliflower bump gets its own lit cap)
    tops = np.full(nwc, 1e9, np.float32)
    slope = np.zeros(nwc, np.float32)
    kr = int(max(1.5 * sc, 1.0))
    for i in range(nwc):
        acc = 0.0
        cnt = 0.0
        for j in range(max(i - kr, 0), min(i + kr + 1, nwc)):
            if top[j] < 1e8:
                acc += top[j]
                cnt += 1.0
        if cnt > 0:
            tops[i] = acc / cnt
    # broad crest (lobe-scale smoothing) for the lit-top value gradient: follows the big lobes, not the
    # notches between the small cauliflower bumps (no pleats / curtain folds)
    topb = np.full(nwc, 1e9, np.float32)
    kb_ = int(max(9.0 * sc, 2.0))
    for i in range(nwc):
        acc = 0.0
        cnt = 0.0
        for j in range(max(i - kb_, 0), min(i + kb_ + 1, nwc)):
            if top[j] < 1e8:
                wj = 1.0 - abs(j - i) / (kb_ + 1.0)
                acc += top[j] * wj
                cnt += wj
        if cnt > 0 and top[i] < 1e8:
            topb[i] = acc / cnt
    # broad facing / steepness from the broad crest (per big lobe, smooth along x)
    ftb = np.zeros(nwc, np.float32)
    stb = np.zeros(nwc, np.float32)
    kd_ = int(max(6.0 * sc, 2.0))
    for i in range(nwc):
        ia_ = max(i - kd_, 0)
        ib_ = min(i + kd_, nwc - 1)
        while ia_ < i and topb[ia_] > 1e8:
            ia_ += 1
        while ib_ > i and topb[ib_] > 1e8:
            ib_ -= 1
        if ib_ > ia_ and topb[i] < 1e8:
            sl = (topb[ib_] - topb[ia_]) / (ib_ - ia_)
            f0 = (sl * lx - ly) / math.sqrt(1.0 + sl * sl)
            f1 = UPW / math.sqrt(1.0 + sl * sl) - 0.08
            ftb[i] = max(f0, f1)
            stb[i] = _ss(1.8, 0.6, abs(sl))
    ftb2 = np.zeros(nwc, np.float32)
    stb2 = np.zeros(nwc, np.float32)
    kq_ = int(max(14.0 * sc, 3.0))
    for i in range(nwc):
        a0_ = 0.0
        a1_ = 0.0
        cnt = 0.0
        for j in range(max(i - kq_, 0), min(i + kq_ + 1, nwc)):
            if topb[j] < 1e8:
                wj = 1.0 - abs(j - i) / (kq_ + 1.0)
                a0_ += ftb[j] * wj
                a1_ += stb[j] * wj
                cnt += wj
        if cnt > 0:
            ftb2[i] = a0_ / cnt
            stb2[i] = a1_ / cnt
    ftb = ftb2
    stb = stb2
    ks_ = int(max(2.0 * sc, 1.0))
    for i in range(nwc):
        ia_ = max(i - ks_, 0)
        ib_ = min(i + ks_, nwc - 1)
        if tops[ia_] < 1e8 and tops[ib_] < 1e8 and ib_ > ia_:
            slope[i] = (tops[ib_] - tops[ia_]) / (ib_ - ia_)
    # crest facing per column, smoothed along the crest (no per-column jitter / drips)
    ft0 = np.zeros(nwc, np.float32)
    for i in range(nwc):
        sl = slope[i]
        ft0[i] = (sl * lx - ly) / math.sqrt(1.0 + sl * sl)
    ftc = np.zeros(nwc, np.float32)
    kf = int(max(2.5 * sc, 1.0))
    for i in range(nwc):
        acc = 0.0
        cnt = 0.0
        for j in range(max(i - kf, 0), min(i + kf + 1, nwc)):
            acc += ft0[j]
            cnt += 1.0
        ftc[i] = acc / cnt
    # back-lit low sun: every exposed TOP also catches the light (facing blended with 'upness')
    for i in range(nwc):
        acc = 0.0
        cnt = 0.0
        for j in range(max(i - kf, 0), min(i + kf + 1, nwc)):
            sl = slope[j]
            acc += 1.0 / math.sqrt(1.0 + sl * sl)
            cnt += 1.0
        ftc[i] = max(ftc[i], UPW * (acc / cnt) - 0.08)
    # crest steepness gate, smoothed along the crest (steep flanks carry no crest light; no per-column
    # jitter -> no vertical striping in the lit caps)
    stc = np.zeros(nwc, np.float32)
    ksm = int(max(6.0 * sc, 2.0))
    for i in range(nwc):
        acc = 0.0
        cnt = 0.0
        for j in range(max(i - ksm, 0), min(i + ksm + 1, nwc)):
            if tops[j] < 1e8:
                acc += _ss(1.5, 0.55, abs(slope[j]))
                cnt += 1.0
        stc[i] = acc / cnt if cnt > 0 else 0.0
    for y in prange(y0, y1):
        for x in range(x0, x1):
            xn = x + noff
            if xn >= PW:
                xn = 2 * PW - 1 - xn
            n1 = N1[y, xn]
            n2 = N2[y, xn]
            n3 = N3[y, xn]
            e1 = -1e9
            nx = 0.0
            ny = -1.0
            ec = -1e9
            ldx = 0.0
            ldy = 0.0
            llev = 0.0
            for li in range(i0, i1):
                cx, cy, rx, ry, wob = Lb[li, 0], Lb[li, 1], Lb[li, 2], Lb[li, 3], Lb[li, 4]
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                r = math.sqrt(dx * dx + dy * dy)
                m = min(rx, ry)
                if (r - 1.0) * m > 4.0 + wob * m:
                    continue
                e = (1.0 - r) * m - r * min(wob * m, 1.2 + 1.2 * sc) * (0.6 * n1 + 0.4 * n2)
                if CAUL and abs(e) < 4.0 * sc + 3.0:
                    e += _cauli(dx, dy, r, m, li, n2, sc)
                e_ = (1.0 - r * (1.0 + wob * 0.4 * n2)) * m
                if e_ > ec:
                    ec = e_
                if e > e1:
                    e1 = e
                    ldx = dx
                    ldy = dy
                    llev = Lb[li, 5]
                    if r > 1e-4:
                        nx = dx / r
                        ny = dy / r
            if e1 < -3.0:
                continue
            fac = nx * lx + ny * ly                     # silhouette facing (for the lost edges only)
            u = (y - ht) / hh + 0.08 * n2 + 0.03 * n1   # 0 at the head top, 1 at its base
            hxn = (x - hx) / hw
            down = _ss(0.3, 0.9, u)
            # lost edges everywhere except the up / sun-facing crest: flanks turned from the sun and the
            # down side dissolve into whatever lies behind (no outline all the way round)
            away = _ss(0.25, -0.5, fac)
            side = _ss(0.5, -0.1, -ny)
            # (head-scale only: per-lobe normals would blur the scallop notches between cauliflower bumps)
            fl_ = _ss(0.25, 1.0, -hxn * lx) * _ss(0.1, 0.5, u)       # flank turned from the sun column
            w = crisp + lost * max(down, 0.35 * fl_)
            a = (e1 + 0.5 * w) / w
            if a <= 0.0:
                continue
            if a > 1.0:
                a = 1.0
            a *= 1.0 - bf * opq * _ss(0.7, 1.15, u + 0.1 * n3)     # base dissolves into the bank below
            if a <= 0.001:
                continue
            # ---- broad FLAT value planes (painted, not airbrushed): a cool body, a lit face plane on the
            # upper sun side, one darker base plane; plane borders wander with the noise (brushed)
            v0 = Hd[hi, H_CORE]
            v1 = Hd[hi, H_CORE + 1]
            v2 = Hd[hi, H_CORE + 2]
            bord = 0.06 * n3 + 0.03 * n2 + 0.012 * n1
            km = mamt * _ss(0.41, 0.33, u - 0.3 * hxn * lx + bord)
            v0 += (Hd[hi, H_MID] - v0) * km
            v1 += (Hd[hi, H_MID + 1] - v1) * km
            v2 += (Hd[hi, H_MID + 2] - v2) * km
            # volume: a warm mauve mid band under the lit top, then a LOST soft falloff into the violet base
            kmv = mauv * _ss(0.16, 0.3, u + bord) * _ss(0.8, 0.62, u + bord)
            v0 += (Hd[hi, H_MVC + 0] - v0) * kmv
            v1 += (Hd[hi, H_MVC + 1] - v1) * kmv
            v2 += (Hd[hi, H_MVC + 2] - v2) * kmv
            ks = (0.35 * _ss(0.45, 0.8, u + bord) + 0.65 * _ss(0.58, 0.66, u + 1.4 * bord)) * 0.95
            if ks > 1.0:
                ks = 1.0
            v0 += (Hd[hi, H_SHD] - v0) * ks
            v1 += (Hd[hi, H_SHD + 1] - v1) * ks
            v2 += (Hd[hi, H_SHD + 2] - v2) * ks
            # ---- cauliflower sub-lobes: every protruding lobe carries its own painted value masses - a
            # lit cap on the side facing the light and a cool shadow crescent under it where it sits
            # on the body (hard seams where one lobe overlaps the next: painted, not airbrushed)
            if llev >= 1.0 and slo > 0.0:
                fd = ldx * slx + ldy * sly + 0.12 * n2
                kcap = slo * 0.0 * _ss(-0.2, 0.9, fd) * (1.0 - 0.6 * _ss(0.2, 0.8, u))
                kund = slo * 0.0 * _ss(-0.1, -1.0, fd) * _ss(-0.1, 0.3, u)
                v0 += (Hd[hi, H_MID] * 1.18 + 0.03 - v0) * kcap
                v1 += (Hd[hi, H_MID + 1] * 1.15 + 0.025 - v1) * kcap
                v2 += (Hd[hi, H_MID + 2] * 1.1 + 0.02 - v2) * kcap
                v0 += (Hd[hi, H_SHD] - v0) * kund
                v1 += (Hd[hi, H_SHD + 1] - v1) * kund
                v2 += (Hd[hi, H_SHD + 2] - v2) * kund
            # ---- crest light: distance to the crest polyline; only crest points facing the sun get it,
            # 1-3 px, widest square to the sun and tapering to nothing as the crest turns away.
            # The same distance drives a broad lit FACE plane hanging under the lit crest: its lower border
            # is an offset of the scalloped crest, so the light reads as cauliflower caps
            kc = 0.0
            rk = 0.0
            fw = hh * 0.13 * (0.25 + 1.8 * _ss(-0.5, 0.6, n3 + 0.7 * n2)) + 1.0
            fwb = min(0.55 * hh, 70.0 * sc) + 3.0
            Rs = int(max(cwp * 3.0 + 4.0, fwb * 1.2 + 2.0 if slo > 0.0 else 0.0))
            dmin = 1e9
            jm = -1
            wsum = 0.0
            fsum = 0.0
            ssum = 0.0
            if lamt > 0.002 or ramt > 0.002 or slo > 0.0:
                for j in range(max(x - x0 - Rs, 0), min(x - x0 + Rs + 1, nwc)):
                    tj = tops[j]
                    if tj > 1e8:
                        continue
                    ddx = float(x - x0 - j)
                    ddy = y - tj
                    d2 = ddx * ddx + ddy * ddy
                    d = math.sqrt(d2)
                    if d < dmin:
                        dmin = d
                        jm = j
                    wj = 1.0 / (1.0 + d2)
                    wsum += wj
                    fsum += ftb[j] * wj
                    ssum += stb[j] * wj
            fts = fsum / wsum if wsum > 0 else 0.0
            sts = ssum / wsum if wsum > 0 else 0.0
            if jm >= 0 and dmin < Rs:
                dp = dmin
                ft = ftc[jm]
                stp = stc[jm]
                if slo > 0.0:
                    ffac = _ss(0.2, 0.8, fts + 0.1 * n3)
                    # a FLAT lit cap with a crisp, scalloped terminator (offset of the crest, broken by noise)
                    # a broad LIT TOP plane: lightest right under the crest, melting down into the mid body
                    # (a value gradient hanging from the crest, not a band with a hard inner border)
                    dv = dp
                    gd = 0.2 * _ss(fwb * (1.0 + 0.3 * n2), 0.0, dv) + 0.35 * _ss(0.8 * fwb, 0.68 * fwb, dv + 0.2 * fwb * n2 + 0.06 * fwb * n1) +                         0.45 * _ss(0.62 * fwb, 0.5 * fwb, dv + 0.18 * fwb * n2 + 0.08 * fwb * n1)
                    kfc = slo * 0.9 * gd * (0.45 + 0.55 * ffac) * (0.75 + 0.25 * sts)
                    fk_ = FACE_K * (0.35 + 0.65 * gd)
                    v0 += (Hd[hi, H_MID] + (Hd[hi, H_LIT] - Hd[hi, H_MID]) * fk_ - v0) * kfc
                    v1 += (Hd[hi, H_MID + 1] + (Hd[hi, H_LIT + 1] - Hd[hi, H_MID + 1]) * fk_ - v1) * kfc
                    v2 += (Hd[hi, H_MID + 2] + (Hd[hi, H_LIT + 2] - Hd[hi, H_MID + 2]) * fk_ - v2) * kfc
                # only crest segments facing up AND toward the sun: steep flanks (running down the sides)
                # and away-facing shoulders get no line at all
                facs = _ss(0.78, 0.97, ft + 0.06 * n3) * stp
                cw = cwp * (0.25 + 0.75 * facs * facs) * (0.55 + 0.6 * _ss(-0.35, 0.45, n3 + 0.7 * n2))
                kc = lamt * _ss(cw + 0.5, cw - 0.4, dp + 0.3 * cw * n1) * facs * _ss(-0.55, -0.05, n2 + 0.6 * n3)
                # the hot crest melts into a warm mid-value over ~20-40 px (light, not an outline)
                wl = min(0.3 * hh, 34.0 * sc) + 2.0
                fw_ = _ss(0.6, 0.95, fts + 0.05 * n3) * sts
                kwm = WARM_K * lamt * fw_ * (0.4 * math.exp(-dp / wl) + 0.6 * _ss(1.05 * wl, 0.85 * wl, dp + 0.3 * wl * n2 + 0.1 * wl * n1)) * (1.0 - kc)
                v0 += ((Hd[hi, H_MID] + Hd[hi, H_LIT]) * 0.5 - v0) * kwm
                v1 += ((Hd[hi, H_MID + 1] + Hd[hi, H_LIT + 1]) * 0.5 - v1) * kwm
                v2 += ((Hd[hi, H_MID + 2] + Hd[hi, H_LIT + 2]) * 0.5 - v2) * kwm
                if ramt > 0.0:
                    fr_ = _ss(0.55, 0.95, ft + 0.05 * n2)
                    rw = (0.7 + 1.6 * fr_ * fr_ * _ss(-0.4, 0.5, n2 + 0.5 * n3)) * sc * dn
                    # hard hot lining broken by small nicks (fine noise), not a continuous contour line
                    rk = ramt * _ss(rw + 0.5, rw - 0.3, dp) * fr_ * stp * _ss(-0.45, 0.1, n3 + 0.6 * n2) * _ss(-0.55, -0.2, n1)
            v0 += (Hd[hi, H_LIT] - v0) * kc
            v1 += (Hd[hi, H_LIT + 1] - v1) * kc
            v2 += (Hd[hi, H_LIT + 2] - v2) * kc
            ia = 1.0 - a
            C[y, x, 0] = C[y, x, 0] * ia + v0 * a
            C[y, x, 1] = C[y, x, 1] * ia + v1 * a
            C[y, x, 2] = C[y, x, 2] * ia + v2 * a
            A[y, x] = A[y, x] + a * (1.0 - A[y, x])
            for c in range(3):
                R[y, x, c] = R[y, x, c] * ia + rk * Hd[hi, H_RIM + c] * a


class Floor:
    def __init__(self, PW, PH, W, hy, sun, sc, y_start, seed=7, size=0.34):
        self.PW, self.PH, self.W, self.hy, self.sun, self.sc = PW, PH, W, hy, sun, sc
        self.y_start = y_start
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.size = size
        self.N1 = _noise(PW, PH, max(PW / (9 * sc), 3), seed + 11, 2, stretch=1.3)
        self.N2 = _noise(PW, PH, max(PW / (40 * sc), 3), seed + 12, 3, stretch=1.6)
        self.N3 = _noise(PW, PH, max(PW / (140 * sc), 3), seed + 13, 3, stretch=2.0)
        # light patches: where the low sun breaks through between banks whole heads light up
        self.NL = _noise(PW, PH, max(PW / (420 * sc), 2), seed + 14, 2, stretch=1.5)

    # --------------------------------------------------------------------------------------- layout
    def rows(self, y_end=None):
        rng = self.rng
        y = self.y_start
        out = []
        if y_end is None:
            y_end = self.PH + 0.02 * self.W
        while y < y_end:
            dy = y - self.hy
            qp = float(np.clip((y - self.y_start) / (self.PH - self.y_start), 0, 1))
            r0 = dy * self.size
            g = GROW[0] + GROW[1] * qp ** GROW[2]       # lobe growth toward the camera (near ~3x)
            out.append((y, r0, g))
            k = rng.uniform(0.3, 0.55) if rng.random() < 0.65 else rng.uniform(0.8, 1.3)
            y += max(r0 * k * (0.8 + 0.5 * qp), 3.0)
        return out

    def _arc(self, lobes, cx, cy, rx, ry, level, cover=1.0):
        """Cover the upper arc of an ellipse with protruding lobes (varied, power-law sizes), recursively:
        a cauliflower silhouette of lobes on lobes."""
        rng = self.rng
        if level > 3:
            return
        fr = (0.2, 0.44) if level == 1 else ((0.24, 0.4) if level == 2 else (0.28, 0.42))
        msz = min(math.sqrt(rx * ry), 1.7 * ry)      # lobe scale from the parent's size, not its (flat) height
        flat = getattr(self, '_flat', 1.0)
        a = -math.pi * rng.uniform(0.96, 1.0)
        a_end = -math.pi * rng.uniform(0.0, 0.04)
        while a < a_end:
            rs = msz * (fr[0] + (fr[1] - fr[0]) * rng.random() ** 1.6)
            if rng.random() < 0.18:
                rs *= 1.55
            rsx = rs * rng.uniform(1.0, 1.45)
            ca, sa = math.cos(a), math.sin(a)
            # lobes on the flanks sit lower and smaller (the crown carries the big ones)
            flank = abs(ca)
            rs_ = rs * (1.0 - 0.35 * flank)
            rsx_ = rsx * (1.0 - 0.35 * flank)
            po = 0.86 if level == 1 else 1.0          # small scallops stand proud of their parent lobe
            px = cx + ca * rx * (po - 0.1 * rng.random())
            py = cy + sa * ry * (po - 0.06 - 0.1 * rng.random()) + (rs * 0.35 * (1 - abs(sa)) if level == 1 else 0.0)
            # cauliflower bumps only where the crest faces the sun (tops / sun-side shoulder); the
            # flank turned away stays a broad smooth curve (few, low, big lobes)
            sdx, sdy = getattr(self, '_sd', (0.0, -1.0))
            fsun = max(ca * sdx + sa * sdy, 0.0)
            cov = cover if level == 1 else cover * (0.75 + 0.25 * fsun)
            rsx_ *= 1.0 / max(flat, 0.2) ** 0.7        # far heads: bumps stretch into flat streaks
            if rng.random() < cov and rs_ > 0.8 * self.sc:
                wob = float(np.clip(0.07 * (6.0 / max(rs_, 6.0)) ** 0.3, 0.02, 0.08))
                ry_ = rs_
                lobes.append((px, py, rsx_, ry_, wob, level))
                if ry_ > 2.8 * self.sc and (level == 1 or fsun > 0.15):
                    self._arc(lobes, px, py, rsx_, ry_, level + 1, cover=0.92)
            # step along the arc ~ one lobe width (in angle)
            circ = math.hypot(rx * sa, ry * ca) + 1e-3
            a += rsx_ * rng.uniform(0.95, 1.5) / circ

    def _sundir(self, x, y):
        dx, dy = self.sun[0] - x, self.sun[1] - y
        n = math.hypot(dx, dy) + 1e-6
        return dx / n, dy / n

    def _head_rec(self, lobes, P, q, lit, rim, hf, bf=0.85, lost=0.3):
        sx = self.sun[0]
        W = self.W
        hx, ht, hh, hw = hf
        L = np.asarray(lobes, F32)
        pad = 6.0
        x0 = float(np.min(L[:, 0] - L[:, 2] * 1.12) - pad)
        x1 = float(np.max(L[:, 0] + L[:, 2] * 1.12) + pad)
        y0 = float(np.min(L[:, 1] - L[:, 3] * 1.12) - pad)
        y1 = float(np.max(L[:, 1] + L[:, 3] * 1.12) + pad)
        # sun-axis proximity (asymmetric: the cool left third falls off fast)
        dxs = (hx - sx) / ((0.17 if hx < sx else 0.27) * W)
        ax = math.exp(-dxs * dxs)
        sunw = ss_py(0.15, 0.65, ax)                             # where crest light exists at all
        sunw *= ss_py(0.31 * W, 0.08 * W, abs(hx - sx))          # none on lobes > ~600 px from the sun column
        lx, ly = self._sundir(hx, ht)
        lx, ly = lx * (0.8 + 1.4 * ax), ly    # crest facing favours the flank turned toward the sun column
        nl_ = math.hypot(lx, ly)
        lx, ly = lx / nl_, ly / nl_
        rec = np.zeros(HF, F32)
        rec[H_X0:H_Y1 + 1] = (x0, y0, x1, y1)
        rec[H_LX], rec[H_LY] = lx, ly
        t = q ** 0.8
        # lit face plane: pinkish gold in the sun column, a lifted cool sky-lit lilac-blue away from it
        mid_cool = _lerp(P['core'], P['mid'], 0.78) * np.array([0.8, 0.92, 1.1], F32)
        mid = _lerp(mid_cool, SCAT_HOT, 0.8 * ax ** 1.5 * (0.4 + 0.6 * lit) * (1.0 - 0.5 * q))
        # right of the sun column the sun-facing bodies stay warm: gold (sun axis) -> rose -> violet (yn_02)
        rs_ = ss_py(0.0, 0.18 * W, hx - sx) * ss_py(0.75 * W, 0.3 * W, hx - sx) * (1.0 - ax)
        mid = _lerp(mid, ROSE, 0.5 * rs_ * (0.4 + 0.6 * lit))
        litc = _lerp(P['lit_cool'], P['lit_hot'], min(ax * 1.4 + 0.35 * rs_, 1.0))
        # away from the sun column the body and shadows turn teal-blue (sky fill), near it violet / pink
        # (the farthest rows keep the deck's violet haze so the two meet without a seam)
        tk = 0.75 * (1.0 - ax) * (0.35 + 0.65 * min(q * 2.5, 1.0))
        core = _lerp(P['core'], np.asarray(P['core'], F32) * np.array([0.62, 0.98, 1.0], F32), tk)
        shd = _lerp(P['shade'], np.asarray(P['shade'], F32) * np.array([0.6, 1.0, 1.0], F32),
                    min(tk * 1.25, 1.0))
        rec[H_CORE:H_CORE + 3] = core
        rec[H_MID:H_MID + 3] = mid
        rec[H_LIT:H_LIT + 3] = litc
        rec[H_SHD:H_SHD + 3] = shd
        rec[H_RIM:H_RIM + 3] = RIM_HOT
        # every head gets a crisp lit top edge (peach / pink away from the sun column, hot gold in it)
        rec[H_LAMT] = (0.45 + 0.55 * lit) * (0.3 + 0.7 * sunw) * self.rng.uniform(0.75, 1.0)
        # sub-lobe value masses; light from up-and-toward-the-sun (tops lit even far off the sun axis)
        slx_ = (self.sun[0] - hx) / self.W * 1.6
        sly_ = -1.0
        nn_ = math.hypot(slx_, sly_)
        rec[H_SLX], rec[H_SLY] = slx_ / nn_, sly_ / nn_
        rec[H_SLO] = 0.55 + 0.35 * lit
        rec[H_MAMT] = (0.55 + 0.4 * lit) * (0.75 + 0.25 * ax) * (1.0 - 0.3 * q)
        rec[H_RAMT] = rim * (0.25 + 0.75 * ax ** 1.4) * max(sunw, 0.45) * (0.8 + 0.4 * q)
        rec[H_LOST] = lost * hh
        rec[H_CW] = (1.4 + 2.2 * q) * self.sc * (0.6 + 0.4 * ax)
        rec[H_KIND] = 0.0
        rec[H_BF] = bf
        rec[H_NOFF] = self.rng.integers(0, self.PW // 2)
        rec[H_HX:H_HW + 1] = (hx, ht, hh, hw)
        rec[H_DN] = 0.7 + 0.6 * t
        # warm mauve mid-value band (warm near the sun column, cooler lilac-mauve away), stronger near
        rec[H_MV] = (0.2 + 0.45 * t) * (0.45 + 0.55 * lit)
        rec[H_MVC:H_MVC + 3] = _lerp(_lerp((0.4, 0.3, 0.52), MAUVE, 0.3 + 0.7 * max(ax, rs_)), P['mid'], 0.35 * (1 - t))
        # nearer heads are solid bodies: their bases barely dissolve (a colour falloff, not transparency)
        rec[H_OPQ] = 1.0 - 0.7 * ss_py(0.15, 0.5, q)
        return rec

    def _head(self, heads, lobes, cx, yb, hw, rh, P, q, lit_h, rim_h, skirt=True):
        rng = self.rng
        dcy = yb - rh * 0.3
        self._sd = self._sundir(cx, dcy - rh)
        mine = [(cx, dcy, hw, rh, 0.02, 0)]
        self._arc(mine, cx, dcy, hw, rh, 1)
        hf = (cx, dcy - rh * 1.05, rh * 1.5, hw)
        i0 = len(lobes)
        lobes += mine
        rec = self._head_rec(mine, P, q, lit_h, rim_h, hf, bf=0.8)
        rec[H_I0], rec[H_I1] = i0, len(lobes)
        heads.append(rec)
        # skirt: lower / nearer sub-heads in front, their crisp tops over this head's cool core; shadowed by
        # the head behind them (dimmer crust, almost no lining)
        if not skirt:
            return
        # small far heads stay one flat mass (no row of tiny stamped caps); the random stream is still
        # consumed so the rest of the layout is unchanged
        ghost = rh < 16.0 * self.sc
        # tiers of lower / nearer sub-heads in front of the head (more tiers on the big near heads); each
        # tier sits lower, is shadowed by the head behind it (dimmer crust, almost no lining)
        ntier = 1 + int(rh > 35 * self.sc) + int(rh > 80 * self.sc)
        for tier in range(ntier):
            if rng.random() > (0.85 if tier == 0 else 0.7):
                continue
            m = int(rng.integers(2, 5))
            for j in range(m):
                u = -0.9 + 1.8 * (j + rng.uniform(0.15, 0.85)) / m
                rs = rh * rng.uniform(0.3, 0.55) * (0.85 ** tier)
                rsx = rs * rng.uniform(1.2, 1.9)
                px = cx + u * hw * 0.82
                py = yb - rs * rng.uniform(0.0, 0.3) + rh * (0.15 + 0.28 * tier)
                sub = [(px, py, rsx, rs, 0.03, 0)]
                self._sd = self._sundir(px, py - rs)
                self._arc(sub, px, py, rsx, rs, 1, cover=0.9)
                i0 = len(lobes)
                lobes += sub
                ax_l = lit_h * rng.uniform(0.25, 0.6) * (0.8 ** tier)
                rec = self._head_rec(sub, P, q, ax_l, rim_h * 0.12 * (0.5 ** tier),
                                     (px, py - rs * 1.25, rs * 2.0, rsx), bf=0.95, lost=0.35)
                # (shadowed by the head behind: a value step, no crest line of its own)
                rec[H_LAMT] *= 0.35
                # a skirt's base sinks into the head behind it: a gentle base value, no dark crescent
                rec[H_SHD:H_SHD + 3] = _lerp(rec[H_CORE:H_CORE + 3], rec[H_SHD:H_SHD + 3], 0.35)
                rec[H_I0], rec[H_I1] = i0, len(lobes)
                if ghost:
                    del lobes[i0:]
                    continue
                heads.append(rec)

    def _row(self, heads, lobes, yb, r0, q, g):
        rng = self.rng
        q = min(q / 0.62, 1.0)                # the visible rows (above the summit lip) span the full palette
        t = q ** 0.8
        P = {k: _lerp(FAR[k], NEAR[k], t) for k in FAR}
        self._flat = 0.22 + 0.78 * q ** 1.3    # far heads much flatter: thin streaks toward the horizon
        # aerial perspective: far rows hazier / lighter, lower contrast
        slab = np.zeros(HF, F32)
        slab[H_KIND] = 1.0
        slab[H_HT], slab[H_HH] = yb + r0 * 0.15, r0
        slab[H_CORE:H_CORE + 3] = P['slab']
        slab[H_LOST] = 1.0 + 0.6 * r0 * (1.0 - q) ** 2
        slab[H_SHD:H_SHD + 3] = P['valley']
        heads.append(slab)
        # MASSES, not a tile of pillows: widths drawn from a wide log-normal (a few broad banks, many
        # small heads), each mass = one broad head + 0-2 shoulder heads, with flat dark valleys between
        hl = []
        rowk = rng.uniform(0.7, 1.35)          # rows differ in height (no identical stacked layers)
        x = -r0 * rng.uniform(0.5, 3.0)
        while x < self.PW + r0 * 3:
            if rng.random() < 0.22 + 0.16 * q:
                x += r0 * g ** 0.5 * rng.uniform(0.6, 3.0)           # a valley gap (the slab shows)
            mw = r0 * g ** 0.55 * float(np.clip(np.exp(rng.normal(0.15, 0.75)), 0.3, 2.8))
            mw = min(mw, self.W * rng.uniform(0.09, 0.16))
            asp = rng.uniform(0.3, 0.55) * (1.0 + 0.25 * min(mw / (r0 * g ** 0.55), 2.0) ** 0.5)
            rh = min(mw * asp, r0 * g ** 0.35 * 1.1) * self._flat * rowk
            if rng.random() < 0.08 + 0.1 * q:                          # a few taller billows
                rh *= rng.uniform(1.6, 2.3)
                mw *= rng.uniform(1.0, 1.3)
            cx = x + mw
            yb_ = yb + r0 * rng.uniform(-0.35, 0.3)
            hl.append((cx, yb_, mw, rh))
            # shoulders: smaller, lower heads merging into the mass on one or both sides
            for sd in (-1, 1):
                if rng.random() < 0.45 and mw > 0.8 * r0:
                    sw = mw * rng.uniform(0.35, 0.6)
                    hl.append((cx + sd * mw * rng.uniform(0.7, 1.0), yb_ + rh * rng.uniform(0.1, 0.35), sw,
                               rh * rng.uniform(0.45, 0.7)))
            x += 2.0 * mw * rng.uniform(0.8, 1.05)
        for i in rng.permutation(len(hl)):
            cx, yb_, hw, rh = hl[i]
            pv = float(self.NL[int(np.clip(yb_, 0, self.PH - 1)), int(np.clip(cx, 0, self.PW - 1))])
            patch = float(np.clip(0.5 + 1.3 * pv, 0.0, 1.0))       # some heads sit in shadow
            patch = max(patch, 0.45 * q)                              # the near banks keep a lit crest
            lit = 0.2 + 0.8 * patch
            rim = patch ** 1.5
            self._head(heads, lobes, cx, yb_, hw, rh, P, q, lit, rim)

    # ------------------------------------------------------------------------------------------- bank
    def bank(self, x0, x1, yb, hmin, hmax, q, seed=5, lit_col=(0.74, 0.62, 0.74), tint=(0.8, 1.0, 1.03)):
        """One far cloud BANK (a row of flat cauliflower heads) between x0 and x1 (plate px) on base line yb:
        a lighter sky-lit top plane with a crisp lit upper edge, a cool body, a base that dissolves (lost)
        into whatever lies below. Returns an rgba plate (straight colour)."""
        PW, PH, sc = self.PW, self.PH, self.sc
        keep = self.rng
        self.rng = rng = np.random.default_rng(seed)
        P = {k: _lerp(FAR[k], NEAR[k], q ** 0.8) * np.asarray(tint, F32) for k in FAR}
        self._flat = 0.8
        heads, lobes = [], []
        x = x0 - rng.uniform(0, 60) * sc
        while x < x1:
            mw = rng.uniform(45, 120) * sc
            tp = ss_py(x1, x1 - 0.35 * (x1 - x0), x + mw)            # heads shrink toward the bank's end
            rh = rng.uniform(hmin, hmax) * sc * (0.45 + 0.55 * tp)
            cx = x + mw
            self._head(heads, lobes, cx, yb + rng.uniform(-0.25, 0.2) * rh, mw, rh, P, q, 0.8, 0.0, skirt=False)
            x += mw * rng.uniform(1.3, 1.9)
        for rec in heads:
            rec[H_LX], rec[H_LY] = 0.5, -0.866            # the crest light on the tops / sun-side shoulders
            rec[H_LIT:H_LIT + 3] = lit_col
            rec[H_LAMT] = 0.75
            rec[H_CW] = 2.2 * sc
            rec[H_MAMT] = 0.8
            rec[H_BF] = 1.0
            rec[H_LOST] = rec[H_HH] * 0.45
        C = np.zeros((PH, PW, 3), F32)
        A = np.zeros((PH, PW), F32)
        R = np.zeros((PH, PW, 3), F32)
        Hd = np.ascontiguousarray(np.stack(heads).astype(F32))
        Lb = np.ascontiguousarray(np.asarray(lobes, F32))
        _paint(C, A, R, Hd, Lb, self.N1, self.N2, self.N3, float(sc))
        C = C / np.maximum(A, 1e-4)[..., None]
        self.rng = keep
        return np.dstack([C, A]).astype(F32)

    # ------------------------------------------------------------------------------------------ paint
    def paint(self, groups=((0.0, 0.3), (0.3, 0.62), (0.62, 1.01)), y_end=None):
        """Returns [(rgba plate, rim plate, depth_q, rows)] per group (far -> near)."""
        PW, PH = self.PW, self.PH
        rows = self.rows(y_end)
        yN = self.PH
        out = []
        ri = 0
        for (g0, g1) in groups:
            C = np.zeros((PH, PW, 3), F32)
            A = np.zeros((PH, PW), F32)
            R = np.zeros((PH, PW, 3), F32)
            qs = []
            heads, lobes = [], []
            while ri < len(rows):
                yb, r0, g = rows[ri]
                q = (yb - self.y_start) / max(yN - self.y_start, 1)
                if q >= g1:
                    break
                qs.append(q)
                self._row(heads, lobes, yb, r0, q, g)
                ri += 1
            if not qs:
                continue
            Hd = np.ascontiguousarray(np.stack(heads).astype(F32))
            Lb = np.ascontiguousarray(np.asarray(lobes, F32))
            _paint(C, A, R, Hd, Lb, self.N1, self.N2, self.N3, float(self.sc))
            C = C / np.maximum(A, 1e-4)[..., None]          # straight colour for the compositor
            out.append((np.dstack([C, A]).astype(F32), R.astype(F32), float(np.mean(qs)), rows))
        return out
