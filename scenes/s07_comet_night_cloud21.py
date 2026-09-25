"""Round-21 night clouds for s07_comet_night (final-panel + reviewer: blotchy lavender paper cut-outs, jaggy
dark-fringed silhouettes, streaky flat interiors, noise cloudlets, a slab pasted on the tail).

yn_04 / yn_05 painted cumulus built as a CAULIFLOWER HIERARCHY of exact circles (big clump lobes -> medium
lobes on their upper arcs -> small florets on top), rasterised at 3x and area-downsampled, so every silhouette
and every internal lobe edge is anti-aliased (no stepped pixels, no matte fringe - the plate is returned
premultiplied).  Painter's order (upper lobes behind, lower lobes in front) gives lobe-on-lobe separation.

Paint (no streak textures):
  * crisp cool-white moonlit plane only on the top crests (dome lambert of the front lobe x height in the mass);
  * bodies fall off softly blue-grey -> navy toward the base; the base is translucent and LOST (feathered
    20-40 px) so the stars show through;
  * a thin warm pink -> cyan rim on the edges that face the comet head.
Only a few big clustered banks - no scattered cloudlets, nothing sitting on the tail.
"""
import math

import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _smooth_noise(w, h, cell, rng):
    gw, gh = max(int(w / cell) + 3, 3), max(int(h / cell) + 3, 3)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(n[:h, :w], (0, 0), cell * 0.25 + 0.5)


def _bank_circles(bank, rng, s):
    """Hierarchical cauliflower lobes for one bank.  Returns (N, 4) array of x, y, r, aspect (plate px)."""
    circ = []
    for ic, (cx, cy, R, flat) in enumerate(bank['clumps']):
        n_before = len(circ)
        nb = int(rng.integers(3, 6))
        bigs = []
        for k in range(nb):
            u = (k / max(nb - 1, 1) - 0.5) * 2 if nb > 1 else 0.0
            r = R * rng.uniform(0.42, 0.62) * (1 - 0.35 * abs(u) ** 1.5)
            x = cx + u * R * 1.25 + rng.normal(0, 0.1 * R)
            y = cy - r * 0.3 + abs(u) * 0.3 * R + rng.normal(0, 0.06 * R)
            bigs.append((x, y, r, rng.uniform(1.25, 1.7)))
        # flat base: wide low lobes under the clump
        for k in range(2):
            u = (k - 0.5) * 1.3
            r = R * rng.uniform(0.3, 0.4) * flat
            bigs.append((cx + u * R, cy + 0.12 * R, r, 2.2))
        circ += bigs
        meds = []
        for (x, y, r, a) in bigs[:nb]:
            nm = int(rng.integers(3, 6))
            for j in range(nm):
                th = math.radians(rng.uniform(-160, -20))
                rm = r * rng.uniform(0.26, 0.42)
                d = r * rng.uniform(0.75, 0.95)
                meds.append((x + math.cos(th) * d * a, y + math.sin(th) * d, rm, rng.uniform(1.0, 1.35)))
        circ += meds
        for (x, y, r, a) in meds:
            if r < 9 * s:
                continue
            for j in range(int(rng.integers(0, 3))):
                th = math.radians(rng.uniform(-155, -25))
                rf = r * rng.uniform(0.3, 0.5)
                d = r * rng.uniform(0.78, 0.95)
                circ.append((x + math.cos(th) * d * a, y + math.sin(th) * d, max(rf, 4.0 * s), rng.uniform(1.0, 1.25)))
        for k in range(n_before, len(circ)):
            circ[k] = tuple(circ[k]) + (ic,)
    return np.array(circ, np.float32)


def _paint_bank(bank, rng, s, comet_xy, ss=3):
    C_ = _bank_circles(bank, rng, s)
    pad = 40 * s + 10
    ext = C_[:, 2] * C_[:, 3]
    x0 = int(math.floor((C_[:, 0] - ext).min() - pad))
    x1 = int(math.ceil((C_[:, 0] + ext).max() + pad))
    y0 = int(math.floor((C_[:, 1] - C_[:, 2]).min() - pad))
    y1 = int(math.ceil((C_[:, 1] + C_[:, 2]).max() + pad))
    w, h = x1 - x0, y1 - y0
    W2, H2 = w * ss, h * ss
    shear = bank.get('shear', 0.12)
    tone = bank.get('tone', 1.0)
    # painterly irregularity: a gentle smooth domain warp applied to every lobe (organic, still exactly AA)
    WX = _smooth_noise(W2, H2, 30 * s * ss, rng) * (2.6 * s * ss) + _smooth_noise(W2, H2, 10 * s * ss, rng) * (0.9 * s * ss)
    WY = _smooth_noise(W2, H2, 30 * s * ss, rng) * (1.8 * s * ss) + _smooth_noise(W2, H2, 10 * s * ss, rng) * (0.8 * s * ss)
    nz1 = _smooth_noise(W2, H2, 34 * s * ss, rng)
    nzb = _smooth_noise(W2, H2, 80 * s * ss, rng)
    yy0, xx0 = np.mgrid[0:H2, 0:W2].astype(np.float32)
    ldx, ldy = 0.3 * bank.get('lx', 0.0), -1.0
    ln = math.hypot(ldx, ldy)
    ldx, ldy = ldx / ln, ldy / ln
    UPM = np.array([0.52, 0.6, 0.86], np.float32) * tone
    MID = np.array([0.31, 0.38, 0.67], np.float32) * tone
    LOW = np.array([0.13, 0.18, 0.43], np.float32)
    NAV = np.array([0.06, 0.1, 0.27], np.float32)
    TOP = np.array([0.8, 0.87, 1.0], np.float32) * tone
    sdf = np.full((H2, W2), 1e4, np.float32)
    col = np.zeros((H2, W2, 3), np.float32)
    # clump by clump, upper (farther) clumps first; each clump is ONE painted value mass
    clumps = bank['clumps']
    corder = np.argsort([c[1] for c in clumps])
    for ic in corder:
        cx_c, cy_c, R, _f = clumps[ic]
        Cc = C_[C_[:, 4] == ic]
        sdc = np.full((H2, W2), 1e4, np.float32)
        for (cx, cy, r, asp, _i) in Cc:
            cx2, cy2, r2 = (cx - x0) * ss, (cy - y0) * ss, r * ss
            rx2 = r2 * asp
            a0, a1 = int(max(cx2 - rx2 - 8 * ss - r2 * shear, 0)), int(min(cx2 + rx2 + 8 * ss + r2 * shear, W2))
            b0, b1 = int(max(cy2 - r2 - 8 * ss, 0)), int(min(cy2 + r2 + 8 * ss, H2))
            if a1 <= a0 or b1 <= b0:
                continue
            dyp = yy0[b0:b1, a0:a1] + 0.5 - cy2 + WY[b0:b1, a0:a1]
            dxp = xx0[b0:b1, a0:a1] + 0.5 - cx2 + dyp * shear + WX[b0:b1, a0:a1]
            d = np.sqrt((dxp / rx2) ** 2 + (dyp / r2) ** 2)
            sdc[b0:b1, a0:a1] = np.minimum(sdc[b0:b1, a0:a1], (d - 1.0) * r2)
        Ac = np.clip(0.5 - sdc / (0.9 * ss), 0, 1)
        # vertical position inside this clump (0 crest .. 1 base)
        gc = np.clip((yy0 / ss + y0 - (cy_c - 0.95 * R)) / (1.35 * R), 0, 1)
        gv = np.clip(gc + 0.07 * nzb, 0, 1)
        cc = UPM + (MID - UPM) * _ss(0.1, 0.5, gv)[..., None]
        cc = cc + (LOW - cc) * _ss(0.4, 0.8, gv)[..., None]
        cc = cc + (NAV - cc) * _ss(0.7, 1.05, gv)[..., None]
        # moonlit plane = offset-mask cel: points whose neighbour TOWARD the moon lies outside the clump.  Its
        # lower boundary is the crest silhouette shifted down -> scalloped like the cauliflower crest.
        sdcc = np.clip(sdc, -80 * ss, 80 * ss)

        def band(delta, soft, var):
            dd = delta * np.clip(1 + var * nz1, 0.25, 2.2)
            sd_s = cv2.remap(sdcc, xx0 + ldx * dd, yy0 + ldy * dd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return np.clip(0.5 + sd_s / soft, 0, 1)
        k2 = band(0.42 * R * ss, 3.5 * ss, 0.3) * (1 - _ss(0.5, 0.85, gv))
        k1 = band(0.15 * R * ss, 0.9 * ss, 0.6)
        cc = cc + (UPM * 1.1 - cc) * (k2 * 0.75)[..., None]
        cc = cc + (TOP - cc) * k1[..., None]
        col = col * (1 - Ac[..., None]) + cc * Ac[..., None]
        sdf = np.minimum(sdf, sdc)
    ins = (sdf < 0).astype(np.float32)
    # vertical position inside the whole MASS (for the lost base), smoothed across lobes/columns
    cum = np.cumsum(ins, 0)
    d_dn = cum[-1:, :] - cum
    sg = 0.03 * h * ss + 2
    wn = cv2.GaussianBlur(ins, (0, 0), sigmaX=sg * 1.6, sigmaY=sg * 0.5) + 1e-4
    d_up_s = cv2.GaussianBlur(cum * ins, (0, 0), sigmaX=sg * 1.6, sigmaY=sg * 0.5) / wn
    d_dn_s = cv2.GaussianBlur(d_dn * ins, (0, 0), sigmaX=sg * 1.6, sigmaY=sg * 0.5) / wn
    g = np.clip(d_up_s / (d_up_s + d_dn_s + 1.0), 0, 1)
    col = col + (NAV - col) * (0.5 * _ss(0.6, 1.0, g))[..., None]
    # outward normal of the silhouette
    sdb = cv2.GaussianBlur(np.clip(sdf, -20 * ss, 20 * ss), (0, 0), 1.5 * ss)
    gx = cv2.Sobel(sdb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sdb, cv2.CV_32F, 0, 1, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = gx / gn, gy / gn
    # ---- alpha: crisp AA on top/sides, lost feathered translucent base (feather follows the smoothed mass)
    A = np.clip(0.5 - sdf / (0.9 * ss), 0, 1)
    down = _ss(0.0, 0.7, ny)
    lost = 26 * s * ss * bank.get('lost', 1.0)
    feather = _ss(0.0, lost * (1.0 + 0.4 * nz1), np.clip(-sdf, 0, None) * 0.5 + d_dn_s * 0.5)
    A = A * (1 - down * (1 - feather) * 0.97) * (1 - 0.4 * _ss(0.55, 1.0, g))
    # thin warm rim on the edges facing the comet head (pink at the edge, cyan just inside)
    cx_, cy_ = (comet_xy[0] - x0) * ss, (comet_xy[1] - y0) * ss
    vx, vy = cx_ - xx0, cy_ - yy0
    vn = np.sqrt(vx * vx + vy * vy) + 1e-3
    face = _ss(0.2, 0.8, (nx * vx + ny * vy) / vn)
    edge_d = -sdf
    rim_w = 2.6 * s * ss
    rk = bank.get('rim', 1.0)
    rim = np.exp(-np.clip(edge_d, 0, None) / rim_w) * (edge_d > -0.5) * face * rk
    cyr = np.exp(-np.clip(edge_d - 2.5 * rim_w, 0, None) / (2.5 * rim_w)) * (edge_d > 2 * rim_w) * face * 0.3 * rk
    col = col + (np.array([0.35, 0.72, 0.95], np.float32) - col) * np.clip(cyr, 0, 1)[..., None] * 0.6
    col = col + (np.array([1.0, 0.64, 0.76], np.float32) - col) * np.clip(rim, 0, 1)[..., None] * 0.85
    # area downsample -> AA silhouette + AA internal edges; premultiplied
    pm = cv2.resize(col * A[..., None], (w, h), interpolation=cv2.INTER_AREA)
    Ad = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    return pm, Ad, x0, y0


def night_clouds21(Wp, Hp, banks, H, comet_xy, tail_glow=None, seed=21):
    """Premultiplied RGBA plate (Hp x Wp) of the near night cloud banks."""
    s = H / 1080.0
    out = np.zeros((Hp, Wp, 4), np.float32)
    for ib, bank in enumerate(banks):
        rng = np.random.default_rng(seed * 100 + ib * 7 + 3)
        pm, A, bx0, by0 = _paint_bank(bank, rng, s, comet_xy)
        h, w = A.shape
        xa, xb = max(bx0, 0), min(bx0 + w, Wp)
        ya, yb = max(by0, 0), min(by0 + h, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        a = A[sl][..., None] * bank.get('amt', 1.0)
        p = pm[sl] * bank.get('amt', 1.0)
        dst = out[ya:yb, xa:xb]
        dst[..., :3] = p + dst[..., :3] * (1 - a)
        dst[..., 3:4] = a + dst[..., 3:4] * (1 - a)
    if tail_glow is not None:
        # the tail's light glances across the cloud tops it streams past (premultiplied add)
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        tg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        out[..., :3] += np.clip(tg, 0, 0.4) * 0.5 * out[..., 3:4]
    return out


def layout21(Wp, W, H, my):
    """Frame fractions at t=0.  (cx, cy, R, flat) per clump: clustered banks streaming toward the comet."""
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    banks = []

    def bank(clumps, **kw):
        cl = []
        for (fx, fy, R, flat) in clumps:
            x, y = fp(fx, fy)
            cl.append((x, y, R * H, flat))
        d = dict(clumps=cl)
        d.update(kw)
        banks.append(d)
    # top-left: big bank partly off the top, broken into clusters that thin out toward the comet
    bank([(0.02, 0.07, 0.11, 1.0), (0.1, 0.1, 0.12, 1.0), (0.2, 0.08, 0.1, 0.9)], lx=1.0, tone=1.0)
    bank([(0.28, 0.17, 0.07, 0.9), (0.335, 0.2, 0.055, 0.8)], lx=1.0, tone=0.96)
    # top-right: bank partly off the top/right edge
    bank([(0.8, 0.07, 0.085, 0.9), (0.88, 0.05, 0.11, 1.0), (0.97, 0.09, 0.1, 1.0)], lx=-1.0, tone=0.98)
    # left edge, lower: a medium cluster that holds the top-left of the frame later in the tilt
    bank([(0.03, 0.38, 0.07, 0.9), (0.11, 0.4, 0.06, 0.8)], lx=1.0, tone=0.9, lost=1.2)
    return banks


def crisp_twilight(pl, s, gain=2.6, lit_px=2.5, lit_col=(0.8, 0.68, 0.9), lit_amt=0.45):
    """Straight-RGBA twilight-row plate -> crisper silhouette (the tops read smeary) + a thin crisp lit crest
    catching the high sky light.  The warm afterglow underside rim painted by cloud20 is kept."""
    out = pl.copy()
    a = np.clip(pl[..., 3], 0, 1)
    ah = np.clip((a - 0.45) * gain + 0.5, 0, 1) * _ss(0.03, 0.2, a)
    out[..., 3] = ah
    k = max(int(round(lit_px * s)), 1)
    band = np.zeros_like(ah)
    for i in range(1, k + 1):
        sh = np.zeros_like(ah)
        sh[i:] = ah[:-i]
        band = np.maximum(band, np.clip(ah - sh, 0, 1) * (1 - (i - 1) / (k + 1)))
    # only on the upper (sky-facing) edges
    rng = np.random.default_rng(5)
    h, w = a.shape
    brk = _ss(-0.6, 0.9, _smooth_noise(w, h, 45 * s + 4, rng))       # broken, not an even outline
    lit = cv2.GaussianBlur(band, (0, 0), 0.5) * lit_amt * (0.15 + 0.85 * brk)
    out[..., :3] = pl[..., :3] * (1 - lit[..., None]) + np.array(lit_col, np.float32) * lit[..., None]
    return out
