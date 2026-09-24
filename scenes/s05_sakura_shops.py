"""s05_sakura round 21: painted ground-floor shopfronts on the far-bank front row (panel: 'the shopfronts at left
are random coloured rectangles; give them real awnings, windows and signage structure').

Works on the supersampled environment buffer in world facade coordinates (along-face metres, height above the
building foot), so every shop moves with the town's exact parallax.  Per building (seeded): a concrete plinth,
2-5 glazed bays in dark aluminium frames (sky reflection gradient at the top of the glass, a dark interior with
warm shelf lights low down, a darker door bay with a handle), a shadow band cast by the awning onto the glass,
a sloped fabric awning (solid or striped, lit top, darker scalloped valance) and a plain fascia board with a
lighter trim.  The facade's own light level is kept (lit faces warm, shade faces cool) and the storefront sinks
into the same aerial haze as the rest of the town.
"""
import numpy as np

AWN = np.array([(0.78, 0.2, 0.18), (0.16, 0.46, 0.34), (0.2, 0.34, 0.66), (0.92, 0.62, 0.2),
                (0.55, 0.22, 0.3), (0.9, 0.86, 0.76), (0.3, 0.3, 0.34)], np.float32)
FASCIA = np.array([(0.95, 0.94, 0.9), (0.2, 0.24, 0.3), (0.86, 0.2, 0.16), (0.96, 0.84, 0.5),
                   (0.18, 0.4, 0.3)], np.float32)


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def shopfronts(out, mat, bid, face, uo, Yo, so, layer, Zo, B, seed=2121, zmax=230.0, haze=(0.8, 0.87, 0.96)):
    rgb = out[..., :3]
    af = np.abs(face)
    nb = len(B)
    bs = np.clip(bid, 0, nb - 1)
    row1 = (B[:, 6] == 0) & (B[:, 1] > -45.0)
    fac = (mat == 5) & (layer == 0) & ((af == 1) | (af == 3)) & row1[bs] & (Zo < zmax)
    h = Yo - B[bs, 2]
    m = fac & (h > -0.05) & (h < 4.05)
    ys, xs = np.nonzero(m)
    if not len(ys):
        return
    rng = np.random.default_rng(seed)
    has = rng.random(nb) < 0.8
    bay = rng.uniform(1.5, 2.3, nb)
    awn_i = rng.integers(0, len(AWN), nb)
    stripe = rng.random(nb) < 0.35
    fas_i = rng.integers(0, len(FASCIA), nb)
    has_f = rng.random(nb) < 0.7
    frame_lt = rng.random(nb) < 0.5
    door_b = rng.integers(0, 3, nb)
    warm = rng.uniform(0.3, 1.0, nb)
    b = bs[ys, xs]
    keep = has[b]
    ys, xs, b = ys[keep], xs[keep], b[keep]
    fc = face[ys, xs]
    a_f = np.abs(fc)
    hh = h[ys, xs]
    along = np.where(a_f == 1, so[ys, xs] - B[b, 4], uo[ys, xs] - B[b, 0])
    width = np.where(a_f == 1, B[b, 5] - B[b, 4], B[b, 1] - B[b, 0])
    c0 = rgb[ys, xs].copy()
    # facade light level per (building, face): mean luminance of its pixels above the shop band
    key = b * 8 + (fc + 4)
    up = fac & (h >= 4.0) & (h < 8.0)
    yu, xu = np.nonzero(up)
    ku = bs[yu, xu] * 8 + (face[yu, xu] + 4)
    lum_u = rgb[yu, xu].mean(-1)
    sm = np.bincount(ku, lum_u, minlength=nb * 8 + 8)
    ct = np.bincount(ku, minlength=nb * 8 + 8)
    lvl = np.where(ct > 0, sm / np.maximum(ct, 1), 0.7)
    lf = np.clip(lvl[key] / 0.78, 0.45, 1.2)[:, None].astype(np.float32)
    shade = np.clip((0.95 - lf[:, 0]) / 0.4, 0, 1)[:, None]
    tint = np.array([1.04, 1.0, 0.94], np.float32) * (1 - shade) + np.array([0.86, 0.9, 1.08], np.float32) * shade
    L = lf * tint
    col = c0.copy()
    # ---- plinth
    pl = hh < 0.3
    col[pl] = (np.array([0.42, 0.42, 0.46], np.float32) * L[pl])
    # ---- glazed bays
    nbay = np.maximum(np.round(width / bay[b]), 1)
    bw = width / nbay
    bi = np.floor(along / bw)
    la = along - bi * bw
    gl = (hh >= 0.3) & (hh < 2.35)
    fr_w = 0.09
    frame = gl & ((la < fr_w) | (la > bw - fr_w) | (hh < 0.38) | (hh > 2.27) |
                  (np.abs(hh - 1.95) < 0.035))                  # transom bar
    fcol = np.where(frame_lt[b][:, None], np.array([0.72, 0.74, 0.78], np.float32),
                    np.array([0.2, 0.21, 0.25], np.float32)).astype(np.float32)
    gy = np.clip((hh - 0.38) / 1.9, 0, 1)[:, None]               # 0 bottom, 1 top of glass
    sky = np.array([0.72, 0.82, 0.92], np.float32)
    dark = np.array([0.46, 0.42, 0.44], np.float32)        # round 22: lit interior value, not a black band
    gcol = dark + (sky - dark) * (gy ** 1.2) * 0.9
    # diagonal sheen stripe across each pane
    sh_ = np.abs(((la * 0.9 + hh * 0.6) % 1.4) - 0.7)
    gcol = gcol + np.array([0.18, 0.2, 0.22], np.float32) * (_ss((0.08 - sh_) / 0.05))[:, None] * gy
    # warm interior: shelf lights low in the glass
    shelf = (np.abs(((hh - 0.4) % 0.45) - 0.22) < 0.03) & (hh < 1.8)
    inner = np.clip(1 - np.abs(la / np.maximum(bw, 1e-3) - 0.5) * 2.2, 0, 1)
    wk = (warm[b] * inner * (1 - gy[:, 0]))[:, None]
    gcol = gcol + np.array([0.6, 0.42, 0.2], np.float32) * wk * 0.8
    gcol = np.where(shelf[:, None], gcol + np.array([0.3, 0.22, 0.1], np.float32) * wk, gcol)
    # door bay: darker glass, a vertical handle
    door = (bi == (door_b[b] % nbay))
    gcol = np.where(door[:, None], gcol * 0.6, gcol)
    handle = door & (np.abs(la - 0.78 * bw) < 0.03) & (np.abs(hh - 1.1) < 0.3)
    gcol = np.where(handle[:, None], np.array([0.75, 0.76, 0.78], np.float32), gcol)
    gcol = np.where(frame[:, None], fcol, gcol)
    gl_ = gl
    col[gl_] = (gcol * np.clip(L * 0.85 + 0.15, 0, 1.3))[gl_]
    # awning cast shadow on the glass top
    sdep = 0.55 * (1 - 0.5 * shade[:, 0])
    ash_k = _ss((hh - (2.35 - sdep)) / 0.12)[:, None] * gl[:, None]
    col = col * (1 - ash_k * (1 - np.array([0.55, 0.58, 0.74], np.float32)))
    # ---- awning (sloped fabric)
    aw = (hh >= 2.35) & (hh < 3.2)
    ac = AWN[awn_i[b]]
    st_on = stripe[b] & ((np.floor(along / 0.32) % 2) == 1)
    ac = np.where(st_on[:, None], np.array([0.95, 0.93, 0.88], np.float32), ac)
    ty = np.clip((hh - 2.35) / 0.85, 0, 1)[:, None]
    ac = ac * (0.72 + 0.4 * ty)                                   # lit toward the top of the slope
    top = (ty > 0.62)                                     # round 22: the sunlit top face of the awning
    ac = np.where(top, np.minimum(ac * 1.3 + 0.1, 1.15) * np.array([1.05, 1.0, 0.9], np.float32), ac)
    ac = ac + (np.minimum(ac * 1.35 + 0.08, 1.2) - ac) * (ty > 0.94)   # crisp lit edge
    val = aw & (hh < 2.35 + 0.2 + 0.05 * np.abs(np.sin(along * np.pi / 0.32)))
    ac = np.where(val[:, None], ac * 0.72, ac)                    # scalloped valance in shade
    col[aw] = (ac * L)[aw]
    # ---- fascia board
    fa = (hh >= 3.2) & (hh < 4.0) & has_f[b]
    fcl = FASCIA[fas_i[b]]
    trim = (hh < 3.26) | (hh > 3.93) | (along < 0.12) | (along > width - 0.12)
    fcl = np.where(trim[:, None], np.minimum(fcl * 0.5 + 0.5, 1.0), fcl)
    fy = np.clip((hh - 3.2) / 0.8, 0, 1)[:, None]
    fcl = fcl * (np.array([0.86, 0.9, 1.04], np.float32) * (1 - fy) + np.array([1.08, 1.02, 0.9], np.float32) * fy)
    fcl = np.where(((hh > 3.26) & (hh < 3.34))[:, None], fcl * 0.7, fcl)      # shadow under the lower trim
    col[fa] = (fcl * L)[fa]
    # ---- round 22: noren curtain in the door bay (split fabric panels, crest mark)
    NOREN = np.array([(0.16, 0.22, 0.44), (0.62, 0.14, 0.14), (0.86, 0.8, 0.66), (0.2, 0.36, 0.3)], np.float32)
    nor_i = rng.integers(0, len(NOREN), nb)
    has_n = rng.random(nb) < 0.65
    lb = la / np.maximum(bw, 1e-3)
    wav = 0.03 * np.sin(la * 18.0)
    nr = door & has_n[b] & (hh > 1.35 + wav) & (hh < 2.3) & (lb > 0.06) & (lb < 0.94)
    slit = (np.abs(((lb - 0.06) / 0.88 * 3.0) % 1.0 - 0.0) < 0.04) & (hh < 2.05)
    ncol = NOREN[nor_i[b]] * (0.8 + 0.25 * np.clip((hh - 1.35) / 0.9, 0, 1))[:, None]
    crest = (np.hypot(lb - 0.5, (hh - 1.75) / np.maximum(bw, 1e-3)) < 0.09) & ~(np.hypot(lb - 0.5, (hh - 1.75) / np.maximum(bw, 1e-3)) < 0.055)
    ncol = np.where(crest[:, None], np.array([0.95, 0.93, 0.88], np.float32), ncol)
    m_ = nr & ~slit
    col[m_] = (ncol * L)[m_]
    # ---- vertical nobori banners standing in front of some shops (pole + fabric, lit / shade halves)
    NOB = np.array([(0.85, 0.16, 0.14), (0.95, 0.93, 0.86), (0.16, 0.34, 0.62), (0.95, 0.72, 0.18)], np.float32)
    nob_n = rng.integers(0, 3, nb)
    for k_ in range(2):
        pos = rng.uniform(0.15, 0.85, nb)
        ci = rng.integers(0, len(NOB), nb)
        on = (k_ < nob_n[b])
        ap = pos[b] * width
        d_ = along - ap
        pole = on & (np.abs(d_) < 0.035) & (hh < 2.75)
        flag = on & (d_ > 0.035) & (d_ < 0.5) & (hh > 0.7) & (hh < 2.6)
        fc_ = NOB[ci[b]] * np.where(d_ < 0.26, 1.0, 0.8)[:, None]
        edge = flag & ((d_ > 0.44) | (hh > 2.52) | (hh < 0.78))
        fc_ = np.where(edge[:, None], np.array([0.95, 0.94, 0.9], np.float32), fc_)
        col[flag] = (fc_ * L)[flag]
        col[pole] = (np.array([0.3, 0.3, 0.33], np.float32) * L)[pole]
    # ---- street clutter against the shopfront: planters, A-frame signs, parked bicycles
    kind = rng.integers(0, 4, (nb, 3))
    cpos = rng.uniform(0.05, 0.8, (nb, 3))
    for k_ in range(3):
        kd = kind[b, k_]
        a0 = cpos[b, k_] * width
        d_ = along - a0
        # planter: box + round green dabs
        pb = (kd == 0) & (d_ > 0) & (d_ < 0.9) & (hh < 0.45)
        col[pb] = (np.array([0.5, 0.36, 0.26], np.float32) * L[pb] * np.where(hh[pb] > 0.38, 1.25, 1.0)[:, None])
        leaf = (kd == 0) & (np.hypot(((d_ % 0.3) - 0.15) / 0.16, (hh - 0.55) / 0.2) < 1.0) & (d_ > 0) & (d_ < 0.9)
        lcol = np.where((hh > 0.58)[:, None], np.array([0.46, 0.64, 0.3], np.float32), np.array([0.2, 0.36, 0.26], np.float32))
        col[leaf] = (lcol * L)[leaf]
        # A-frame sign: dark board, warm frame, chalk lines
        tap = 0.08 * (hh / 0.95)
        af_ = (kd == 1) & (d_ > tap) & (d_ < 0.6 - tap) & (hh < 0.95)
        brd = (d_ > tap + 0.06) & (d_ < 0.54 - tap) & (hh > 0.35) & (hh < 0.88)
        chalk = brd & (np.abs(((hh - 0.4) % 0.12) - 0.06) < 0.012) & (d_ > 0.15) & (d_ < 0.45)
        acol = np.where(brd[:, None], np.array([0.14, 0.17, 0.16], np.float32), np.array([0.62, 0.45, 0.3], np.float32))
        acol = np.where(chalk[:, None], np.array([0.9, 0.88, 0.82], np.float32), acol)
        col[af_] = (acol * L)[af_]
        # bicycle: two wheels (rings), frame lines, saddle
        wr = 0.3
        w1 = np.hypot(d_ - 0.32, hh - wr)
        w2 = np.hypot(d_ - 1.34, hh - wr)
        ring = (np.abs(w1 - wr) < 0.035) | (np.abs(w2 - wr) < 0.035)
        def seg(ax, ay, bx, by, t=0.03):
            vx, vy = bx - ax, by - ay
            u = np.clip(((d_ - ax) * vx + (hh - ay) * vy) / (vx * vx + vy * vy), 0, 1)
            return np.hypot(d_ - ax - u * vx, hh - ay - u * vy) < t
        frm = seg(0.32, wr, 0.75, wr) | seg(0.75, wr, 1.2, 0.72) | seg(0.62, 0.72, 1.2, 0.72) |             seg(0.32, wr, 0.62, 0.72) | seg(0.75, wr, 0.62, 0.8) | seg(1.2, 0.72, 1.34, wr) | seg(0.55, 0.82, 0.72, 0.82, 0.04)
        bk = (kd == 2) & (ring | frm) & (d_ > -0.1) & (d_ < 1.8)
        bcol = np.where(rng.random() < 0.5, np.array([0.18, 0.2, 0.26], np.float32), np.array([0.62, 0.2, 0.18], np.float32))
        col[bk] = (bcol * L * 0.9)[bk]
    # ---- aerial haze (same law as the town shading)
    Z = Zo[ys, xs]
    k = (0.5 * (1 - np.exp(-Z / 260.0)))[:, None]
    col = col + (np.array(haze, np.float32) - col) * k
    rgb[ys, xs] = col


def calm_windows(out, mat, bid, face, Yo, layer, Zo, B, zmax=230.0):
    """upper-floor windows on the front row: the saturated blue glare panes and flat yellow panels were reading
    as random coloured rectangles -> glass pulled to a calm painted sky reflection (lighter at the top of each
    pane run, darker low), yellow panels to a muted warm curtain."""
    import cv2
    rgb = out[..., :3]
    af = np.abs(face)
    nb = len(B)
    bs = np.clip(bid, 0, nb - 1)
    row1 = (B[:, 6] == 0) & (B[:, 1] > -45.0)
    h = Yo - B[bs, 2]
    m = (mat == 5) & (layer == 0) & ((af == 1) | (af == 3)) & row1[bs] & (Zo < zmax) & (h >= 4.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = np.clip((b - r - 0.12) / 0.2, 0, 1) * m
    yel = np.clip(((r + g) * 0.5 - b - 0.22) / 0.15, 0, 1) * m
    blue = cv2.GaussianBlur(blue.astype(np.float32), (0, 0), 0.8)
    yel = cv2.GaussianBlur(yel.astype(np.float32), (0, 0), 0.8)
    lum = rgb.mean(-1, keepdims=True)
    glass = np.array([0.5, 0.6, 0.72], np.float32) * (0.75 + 0.35 * np.clip(lum, 0, 1.2))
    cur = np.array([0.86, 0.78, 0.64], np.float32) * np.clip(lum * 1.1, 0.5, 1.0)
    rgb[:] = rgb + (glass - rgb) * (0.7 * blue)[..., None]
    rgb[:] = rgb + (cur - rgb) * (0.6 * yel)[..., None]
