"""s01 helper (round 4): value repaint of the hero cumulonimbus (geometry from s01_summer_sky_cb5,
painted with lib/clouds2).

Mass-by-mass painting (finish2): every painted mass gets a lit cap facing the key whose terminator is its
own cauliflower silhouette shifted away from the sun (scalloped, soft), a warm mid plane, then a deep cool
blue-grey shadow (#66739F..#8795C0); the depth of the lit cap follows the tower's big form (whole left
flank lit, right flank in shadow). Front masses throw a short cast on the mass behind, crisp at the front
mass's edge. Values are blurred inside each mass only, so every overlap edge stays crisp; small spots and
narrow slivers (drips / fingers) are painted out. Then: a crisp 2-4 px gold/silver lining on the sun-facing
silhouette, a torn anvil tip, and a torn, hazy base that dissolves into the horizon."""
import numpy as np
import cv2

from lib import clouds2 as K
import s01_summer_sky_cb5 as G5


def _c(h):
    return K._c(h)


LIT_HOT = np.array([1.1, 1.08, 1.02], np.float32)      # nearest the sun (HDR, warm)
LIT = _c('#fffdf6') * 1.07                              # brilliant warm white
LIT_LOW = _c('#f4f2ee')                                   # lower lit faces (a touch of air)
MID_W = _c('#e6dcd8')                                     # warm mid plane between lit and shadow
MID_C = _c('#aeb5d3')                                     # cool half-tone where the form turns away
SH_TOP = _c('#66739f')                                    # core shadow high up under the anvil
SH = _c('#7280ae')                                        # shadow plane
SH_LOW = _c('#8795c0')                                    # lower shadow: reflected skylight / haze
SH_REFL = _c('#8a97c2')                                   # a shadow plane catching reflected light
LIT_STEP = _c('#e8e3e3')                                  # lit-on-lit: a back mass just behind a front one                                    # lower shadow: reflected skylight / haze
RIM_GOLD = np.array([1.12, 0.98, 0.72], np.float32)
RIM_SILVER = np.array([1.2, 1.15, 1.02], np.float32)


def _rs(img, w, h, interp=cv2.INTER_AREA):
    return cv2.resize(np.ascontiguousarray(img, np.float32), (w, h), interpolation=interp)


def _despeck(m, min_area, blob_area=0):
    """Remove islands / fill holes smaller than min_area px, and compact round blobs (spots, not
    crescents) smaller than blob_area."""
    m8 = (m > 0.5).astype(np.uint8)
    for val in (1, 0):
        src = m8 if val else 1 - m8
        n, lab, st, _ = cv2.connectedComponentsWithStats(src, connectivity=4)
        small = np.zeros(n, bool)
        ar = st[1:, cv2.CC_STAT_AREA].astype(np.float32)
        bw = st[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
        bh = st[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        blob = (ar < blob_area) & (ar / (bw * bh) > 0.45) & (np.maximum(bw, bh) / np.minimum(bw, bh) < 2.5)
        small[1:] = (ar < min_area) | blob
        m8[small[lab]] = 1 - val
    return m8.astype(np.float32)


def _mblur(F, M, sigma):
    """Blur F inside mask M only (normalised convolution): soft within a mass, never across its edge."""
    Mf = M.astype(np.float32)
    num = cv2.GaussianBlur(F * Mf, (0, 0), sigma)
    den = cv2.GaussianBlur(Mf, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _sample(M, xs, ys):
    return cv2.remap(M, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


Q = dict(kd=(-0.66, -0.75), cap0=0.22, cap1=0.85, mid_f=0.62, cast=0.06, gl_lo=-0.55, gl_hi=0.2,
         gl_sig=0.1, soft_self=3.0, soft_cast=2.0, anvil_cap=0.1, lc_lit=0.7, cap_nz=0.25, mid_nz=0.3, kcast=(-0.66, -0.75))


def finish2(rgba, sun, info, W, haze_col='#d4ebf8', dbg=None, p=None):
    """Mass-by-mass value painting (see module doc): every mass gets a lit cap facing the key whose
    lower-right terminator is its own cauliflower silhouette shifted away from the sun (scalloped, soft),
    a warm mid band, then a deep cool shadow; front masses throw a short cast on the mass behind (crisp
    at the front mass's edge). Values are blurred INSIDE each mass only, so every overlap stays crisp."""
    p = dict(Q, **(p or {}))
    u = W / 1920.0
    top, by, masks, ids = info['top'], info['by'], info['masks'], info['ids']
    Hc = by - top
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    h, w = A.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = (ys - top) / Hc
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    kd = np.array(p['kd'], np.float32)
    kd /= float(np.hypot(*kd))
    # big-form light: the whole tower's body turning away from the key (right / underneath)
    q = 4
    Aq = _rs(A, w // q, h // q)
    Ab = cv2.GaussianBlur(Aq, (0, 0), p['gl_sig'] * Hc / q)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl_ = np.sqrt(gx * gx + gy * gy)
    edge = K._ss(0.0005, 0.01, gl_)
    ndl = (-gx * kd[0] - gy * kd[1]) / (gl_ + 1e-6) * edge
    glq = K._ss(p['gl_lo'], p['gl_hi'], ndl)
    glq = cv2.GaussianBlur(glq, (0, 0), 0.04 * Hc / q)
    GL = _rs(glq, w, h, cv2.INTER_LINEAR)
    ids_ = ids.astype(np.int32).copy()
    ids_[(ids_ < 0) & (A > 0.02)] = 0              # fibres / fringe: part of the anvil
    order = sorted(masks)
    Mf = {i: masks[i].astype(np.float32) for i in order}
    # union of every mass in front of i
    front = {}
    acc = np.zeros((h, w), np.float32)
    for i in reversed(order):
        front[i] = acc.copy()
        acc = np.maximum(acc, Mf[i])
    # slow variation of the lit-cap depth along the masses (the terminator is not an offset outline)
    nzc = K._noise(w // q, h // q, 7.0, 55, 3)
    nzc = _rs((nzc - nzc.mean()) / (nzc.std() + 1e-6), w, h, cv2.INTER_LINEAR)
    nzm = K._noise(w // q, h // q, 12.0, 57, 2)
    nzm = _rs((nzm - nzm.mean()) / (nzm.std() + 1e-6), w, h, cv2.INTER_LINEAR)
    L = np.ones((h, w), np.float32)
    TONE = np.zeros((h, w), np.float32)
    for n_, i in enumerate(order):
        R = ids_ == i
        if not R.any():
            continue
        area = float(Mf[i].sum())
        size = np.sqrt(area)
        cap = size * (p['cap0'] + p['cap1'] * GL) * (1 + p['cap_nz'] * nzc)
        k_ = kd
        if i == 0:                                   # anvil: a lit top, thin cool shadow band under it
            cap = np.full_like(GL, p['anvil_cap'] * Hc)
            k_ = np.array([-0.25, -0.97], np.float32)
        sh = _sample(Mf[i], xs + cap * k_[0], ys + cap * k_[1])
        mf = p['mid_f'] * (1 + p['mid_nz'] * nzm)
        md = _sample(Mf[i], xs + mf * cap * k_[0], ys + mf * cap * k_[1])
        Ls = 1 - 0.5 * md - 0.5 * sh                 # 1 lit, .5 mid, 0 shadow
        Ls = _mblur(Ls, R, p['soft_self'] * u)
        if i > 0:
            c = p['cast'] * Hc
            kc = np.array(p['kcast'], np.float32)
            kc /= float(np.hypot(*kc))
            ca = np.maximum(_sample(front[i], xs + c * kc[0], ys + c * kc[1]),
                            _sample(front[i], xs + 0.4 * c * kc[0], ys + 0.4 * c * kc[1]))
            ca = _mblur(ca, R, p['soft_cast'] * u)
            lc = np.where(GL > 0.62, p['lc_lit'], np.where(GL > 0.32, 0.5, 0.0)).astype(np.float32)
            lc = _mblur(lc, R, 6 * u)
            Ls = np.minimum(Ls, 1 - ca * (1 - lc))
        L[R] = Ls[R]
        TONE[R] = ((n_ * 7) % 5) / 4.0 - 0.5
    # narrow slivers of a back mass peeking between two front masses read as drips / fingers: they take
    # the value of the masses around them
    ko = int(round(0.038 * Hc)) | 1
    ello = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
    sliver = np.zeros((h, w), np.uint8)
    for i in order:
        R8 = (ids_ == i).astype(np.uint8)
        if R8.any():
            sliver |= R8 & (1 - cv2.morphologyEx(R8, cv2.MORPH_OPEN, ello))
    sliver &= (cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5) > 0.02 * Hc).astype(np.uint8)
    if sliver.any():
        keep = 1 - cv2.dilate(sliver, np.ones((3, 3), np.uint8))
        Lk = _mblur(L, keep.astype(bool), 0.015 * Hc)
        sb = cv2.GaussianBlur(sliver.astype(np.float32), (0, 0), 1.5 * u)
        L = L * (1 - sb) + Lk * sb
    # no small isolated value spots: every painted plane is a big mass (spots are filled with the value
    # around them, including their soft fringe)
    ma = int((0.055 * Hc) ** 2)
    kf = np.ones((int(8 * u) * 2 + 1,) * 2, np.uint8)
    for th in (0.9, 0.65, 0.3):
        m = (L > th).astype(np.float32)
        m2 = _despeck(m, ma if th < 0.9 else ma // 3, blob_area=ma * 3)
        for val, tgt in ((1, lambda L_: np.maximum(L_, min(th + 0.1, 1.0) if th < 0.9 else 1.0)),
                         (0, lambda L_: np.minimum(L_, th - 0.1))):
            fill = ((m2 != m) & ((m2 > 0.5) == bool(val))).astype(np.uint8)
            if fill.any():
                fb = cv2.GaussianBlur(cv2.dilate(fill, kf).astype(np.float32), (0, 0), 3 * u)
                L = L * (1 - fb) + tgt(L) * fb
    L = np.where(ids_ >= 0, L, 1.0)
    if dbg is not None:
        dbg.update(L=L, lit=GL)
    # ---- colours
    warm = np.exp(-(dsun / 0.45) ** 2)[..., None]
    low = K._ss(0.3, 0.75, yn)[..., None]
    lit_c = LIT * (1 - low) + LIT_LOW * low
    lit_c = lit_c * (1 - warm) + LIT_HOT * warm
    GLc = GL[..., None]
    mw = 0.55 + 0.45 * K._ss(0.1, 0.55, GLc)
    mid_c = MID_W * mw + MID_C * (1 - mw)
    g = K._ss(0.05, 0.6, yn)[..., None]
    gl2 = K._ss(0.45, 0.8, yn)[..., None]
    sh_c = (SH_TOP * (1 - g) + SH * g) * (1 - gl2) + SH_LOW * gl2
    sh_c = sh_c * (1 + 0.06 * TONE[..., None])
    Lc = L[..., None]
    step_c = LIT_STEP * (1 - warm) + LIT * warm * 0.97
    col = np.where(Lc < 0.5, sh_c * (1 - Lc * 2) + mid_c * (Lc * 2),
                   np.where(Lc < 0.8, mid_c * (1 - (Lc - 0.5) / 0.3) + step_c * ((Lc - 0.5) / 0.3),
                            step_c * (1 - (Lc - 0.8) / 0.2) + lit_c * ((Lc - 0.8) / 0.2)))
    return _post(col.astype(np.float32), A, sun, info, W, haze_col, ys, xs, yn, dsun, kd)


def _streaks(col, A, xa, long_, top, Hc, u, xs, ys, seed=31, n=11):
    rng = np.random.default_rng(seed)
    h, w = A.shape
    tip = xa + long_
    x0b, x1b = int(max(xa + 0.3 * long_, 0)), int(min(tip + 0.9 * long_, w))
    y0b, y1b = int(max(top - 0.02 * Hc, 0)), int(min(top + 0.3 * Hc, h))
    X = xs[y0b:y1b, x0b:x1b]
    Y = ys[y0b:y1b, x0b:x1b]
    acc_a = np.zeros(X.shape, np.float32)
    acc_c = np.zeros(X.shape + (3,), np.float32)
    lit = np.array([1.02, 1.0, 0.97], np.float32)
    cool = K._c('#b9c3e2')
    for k in range(n):
        sx = tip - rng.uniform(0.05, 0.5) * long_
        sy = top + rng.uniform(0.07, 0.2) * Hc
        L = rng.uniform(0.25, 0.75) * long_
        ang = np.radians(rng.uniform(-7, 1))
        bend = rng.uniform(-0.03, 0.03) * Hc
        wd = rng.uniform(0.003, 0.011) * Hc
        op = rng.uniform(0.35, 0.85)
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        s_ = ((X - sx) * d[0] + (Y - sy) * d[1]) / L
        q = (X - sx) * -d[1] + (Y - sy) * d[0] - bend * np.clip(s_, 0, 1) ** 2
        env = np.clip(s_, 0, 1)
        prof = np.sin(np.pi * np.clip(s_, 0, 1) ** 0.6)                 # swells early, long thin taper
        wdt = wd * (0.25 + 0.75 * prof)
        ph = rng.uniform(0, 6.28)
        brk = 0.6 + 0.4 * np.sin(s_ * rng.uniform(9, 16) + ph)             # a few breaks along the stroke
        a = np.exp(-(q / np.maximum(wdt, 0.5 * u)) ** 2) * prof * brk * op * (s_ > 0) * (s_ < 1)
        # fibre grain inside the stroke
        a *= 0.8 + 0.2 * np.sin(q / (0.6 * u + 0.15 * wd) + s_ * 3 + ph)
        t_ = np.clip((sy - top) / (0.2 * Hc), 0, 1)
        c0 = lit * (1 - 0.55 * t_) + cool * (0.55 * t_)             # lower strokes sit in the anvil's shade
        c = c0 * (1 - 0.25 * env[..., None]) + cool * (0.25 * env[..., None])
        acc_c = acc_c * (1 - a[..., None]) + c * a[..., None]
        acc_a = acc_a + a * (1 - acc_a)
    Ar = A[y0b:y1b, x0b:x1b]
    An = Ar + acc_a * (1 - Ar)
    cr = col[y0b:y1b, x0b:x1b]
    col = col.copy()
    col[y0b:y1b, x0b:x1b] = (cr * Ar[..., None] + acc_c / np.maximum(acc_a, 1e-4)[..., None] *
                              (acc_a * (1 - Ar))[..., None]) / np.maximum(An, 1e-4)[..., None]
    A = A.copy()
    A[y0b:y1b, x0b:x1b] = An
    return col, A


def _post(col, A, sun, info, W, haze_col, ys, xs, yn, dsun, kd):
    """Torn anvil tip, the sun-facing lining (+ a tight halation), and the torn hazy base."""
    u = W / 1920.0
    xa, long_, top, by = info['xa'], info['long_'], info['top'], info['by']
    Hc = by - top
    h, w = A.shape
    # ---- anvil tip: the wedge tapers and tears into wind-sheared fibres
    st = K._noise(w, h, 14.0, 77, 3, stretch=2.5)
    st = (st - st.mean()) / (st.std() + 1e-6)
    tipk = (1 - K._ss(top + 0.25 * Hc, top + 0.32 * Hc, ys))
    edge_x = xa + (0.72 + 0.1 * st) * long_ - 0.25 * (ys - top - 0.1 * Hc)
    A = A * (1 - tipk * K._ss(edge_x - 0.02 * Hc, edge_x + 0.1 * Hc, xs))
    # ---- streaming cirrus off the anvil tip: tapered, wind-sheared strokes of varied length / width
    col, A = _streaks(col, A, xa, long_, top, Hc, u, xs, ys)
    # ---- torn, hazy base: aerial perspective, then a ragged dissolve with strands trailing lower right
    hz = K._c(haze_col)
    warmhz = hz * 0.6 + np.array([0.97, 0.96, 0.94], np.float32) * 0.4
    ka = (K._ss(0.5, 0.84, yn) ** 1.2)[..., None]
    lum = col.mean(-1, keepdims=True)
    keepv = (1 - K._ss(0.55, 0.8, yn))[..., None]
    c2 = warmhz + (lum - 0.8) * 0.45 * keepv
    col = col * (1 - 0.85 * ka) + c2 * (0.85 * ka)
    # ragged bottom: a lumpy base line (a few deep tears, lobes hanging lower toward the right, where
    # the wind drags the foot), a tattered edge above it, a few thin strands trailing to the lower right
    xn = (xs[0] - xa) / Hc
    pr = K._noise(w, 8, 6.0, 97, 3)[3]
    pr = (pr - pr.mean()) / (pr.std() + 1e-6)
    pr2 = K._noise(w, 8, 17.0, 98, 2)[3]
    pr2 = (pr2 - pr2.mean()) / (pr2.std() + 1e-6)
    yb = top + (0.73 + 0.045 * pr + 0.015 * pr2 + 0.04 * np.clip(xn, -0.2, 0.6)) * Hc
    n1 = K._noise(w, h, 12.0, 91, 3, stretch=3.5)
    n1 = (n1 - n1.mean()) / (n1.std() + 1e-6)
    dyb = (ys - yb[None, :]) / (0.05 * Hc)
    cut = dyb + 0.5 * n1
    body = K._ss(0.35, -0.15, cut)
    # strands: long thin wisps sheared toward the lower right, below the torn edge
    sh = np.clip(ys - yb[None, :], 0, None) * 1.6
    xs2 = (xs - sh).astype(np.float32)
    n3 = K._noise(w, h, 9.0, 95, 3, stretch=9.0)
    n3 = cv2.remap(n3.astype(np.float32), xs2, ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n3 = (n3 - n3.mean()) / (n3.std() + 1e-6)
    src = cv2.GaussianBlur((A > 0.5).astype(np.float32), (0, 0), 0.02 * Hc)
    src = cv2.remap(src, xs2, ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    strands = K._ss(0.8, 1.5, n3) * K._ss(-0.2, 0.3, dyb) * K._ss(2.6, 0.6, dyb) * np.clip(src * 1.5, 0, 1) * 0.7
    # the flat bottoms of the low masses are torn too (no smooth envelope bases)
    Ab0 = cv2.GaussianBlur(A, (0, 0), 0.02 * Hc)
    gyb = cv2.Sobel(Ab0, cv2.CV_32F, 0, 1, ksize=3)
    gxb = cv2.Sobel(Ab0, cv2.CV_32F, 1, 0, ksize=3)
    down = K._ss(0.3, 0.8, -gyb / (np.sqrt(gxb * gxb + gyb * gyb) + 1e-6)) * K._ss(1e-4, 2e-3, np.abs(gyb))
    down = cv2.GaussianBlur(down.astype(np.float32), (0, 0), 0.01 * Hc) * K._ss(0.5, 0.62, yn)
    dIn = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / (0.035 * Hc)
    n4 = K._noise(w, h, 30.0, 92, 3, stretch=3.0)
    n4 = (n4 - n4.mean()) / (n4.std() + 1e-6)
    tat = K._ss(0.2, 0.7, dIn + 0.45 * n4)
    A = A * (1 - np.clip(down * 1.5, 0, 1) * (1 - tat))
    A = np.maximum(A * body, strands * K._ss(0.0, 0.5, src))
    A = A * (1 - K._ss(0.88, 0.93, yn))
    # the foot melts: overall alpha thins out toward the base line
    A = A * (1 - 0.2 * K._ss(-3.0, 0.0, dyb))
    # ---- the lining: a crisp 2-4 px gold / silver line on the sun-facing silhouette (upper-left flank,
    # crest, anvil top), hottest nearest the sun
    Ab = cv2.GaussianBlur(A, (0, 0), 7 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl
    face = K._ss(-0.15, 0.5, nx * kd[0] + ny * kd[1])
    up = K._ss(-0.3, 0.5, -ny)
    near = np.exp(-(dsun / 0.3) ** 2)
    vz = 1 - K._ss(0.52, 0.7, yn)
    k = np.clip(np.maximum(face, 0.7 * up) * vz * (0.55 + 0.45 * np.exp(-dsun / 0.35)), 0, 1)
    k = np.maximum(k, near * 0.9)
    k *= 1 - 0.85 * K._ss(xa + 0.5 * long_, xa + 0.95 * long_, xs)     # not on the fibres
    dI = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    width = (1.6 + 1.6 * near) * u
    ring = K._ss(0.0, 0.7 * u, dI) * K._ss(width + 0.8 * u, width - 0.3 * u, dI)
    kk = np.clip(ring * k, 0, 1)[..., None]
    gw = np.clip(0.1 + near * 1.1, 0, 1)[..., None]
    rim_c = RIM_SILVER * (1 - gw) + RIM_GOLD * gw
    # a hair of cooler value just inside the line so it reads as light catching the edge
    # the body just inside the line is held a step down (warm pearl) so the line reads as light catching
    # the edge
    band = (K._ss(width + 0.2 * u, width + 1.5 * u, dI) * K._ss(16 * u, 7 * u, dI) * k)[..., None]
    band = cv2.GaussianBlur(band[..., 0], (0, 0), 1.2 * u)[..., None]
    col = col * (1 - 0.1 * band) + np.array([0.9, 0.9, 0.92], np.float32) * (0.1 * band)
    col = col * (1 - kk) + rim_c * kk
    # the line continues a hair OUTSIDE the silhouette (light wrapping the edge) + a tight halation /
    # bloom into the sky, gold nearest the sun
    dO = cv2.distanceTransform((A <= 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    kb_ = cv2.GaussianBlur(k, (0, 0), 2 * u)
    core_o = K._ss(1.3 * u, 0.4 * u, dO) * 0.75
    glow_o = (np.exp(-dO / (3.0 * u)) * 0.5 + np.exp(-dO / (9.0 * u)) * 0.22) * (0.5 + 0.5 * near)
    hk = np.clip(np.maximum(core_o, glow_o) * kb_ * (1 - A), 0, 1)
    A_new = np.clip(A + hk, 0, 1)
    halo_c = rim_c * np.array([1.0, 0.98, 0.9], np.float32)
    col = (col * A[..., None] + halo_c * hk[..., None]) / np.maximum(A_new, 1e-4)[..., None]
    col = np.where(A_new[..., None] > 1e-4, col, halo_c)
    return np.concatenate([col, A_new[..., None]], -1).astype(np.float32)


def tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8'):
    rgba, sun, info = G5.raw_tower(pw, ph, cx, base_y, Hc, W, seed=seed, haze_col=haze_col)
    return finish2(rgba, sun, info, W, haze_col=haze_col), sun
