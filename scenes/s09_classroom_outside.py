"""Outside-the-window plates for s09_classroom.

sky plate  : gnomonic projection onto a plane at unit distance beyond the window wall:
             u = dx / -dz, v = dy / -dz (ray direction d), so the sky sits at infinity.
tree plate : RGBA texture on the plane z = TREE_Z (world metres) - real parallax as the camera moves.
"""
import math
import numpy as np
import cv2

from lib import core as C, sky as S

U0, U1, V0, V1 = -1.2, 3.2, -0.35, 1.05          # sky plate extent in gnomonic units
TX0, TX1, TY0, TY1 = -14.0, 34.0, -7.0, 9.0       # tree plate extent in metres (x, y) on z = TREE_Z
TREE_Z = -10.0


def sky_plate(ppu, sun_uv, seed=4):
    """(H, W, 3) HDR sky: gradient + painted cumulus + distant town/hills near the horizon."""
    pw, ph = int((U1 - U0) * ppu), int((V1 - V0) * ppu)
    hzp = V1 / (V1 - V0)                                     # horizon (v = 0) as fraction of plate height
    sun_p = ((sun_uv[0] - U0) * ppu, (V1 - sun_uv[1]) * ppu)
    # custom gradient in v (the window only shows v ~ -0.05 .. 0.45, so the blue must deepen quickly)
    vv = V1 - (np.arange(ph, dtype=np.float32) + 0.5) / ppu
    stops_v = np.array([-0.4, -0.02, 0.0, 0.03, 0.08, 0.16, 0.28, 0.45, 0.7, 1.1], np.float32)
    stops_c = np.array([[0.62, 0.74, 0.8], [0.78, 0.9, 0.95], [0.86, 0.95, 0.98], [0.74, 0.9, 0.99],
                        [0.52, 0.78, 0.98], [0.33, 0.62, 0.95], [0.2, 0.47, 0.9], [0.12, 0.35, 0.82],
                        [0.07, 0.25, 0.72], [0.04, 0.18, 0.62]], np.float32)
    prof = np.stack([np.interp(vv, stops_v, stops_c[:, c]) for c in range(3)], -1)
    sky = np.repeat(prof[:, None, :], pw, 1).astype(np.float32)
    xs, ys = C.grid(pw, ph)
    d = np.sqrt((xs - sun_p[0]) ** 2 + (ys - sun_p[1]) ** 2) / ppu
    g = np.exp(-d / 0.06) * 0.7 + np.exp(-d / 0.3) * 0.16
    sky = C.screen(sky, np.array([1.0, 0.9, 0.72], np.float32) * np.clip(g, 0, 0.9)[..., None])
    # the sun itself (HDR disc + tight halo) - bloom/flare pick it up
    sun_r = 0.011
    disc = C.smoothstep(sun_r * 1.1, sun_r * 0.9, d)
    sky = sky + (disc * 6.0 + np.exp(-d / 0.02) * 1.2)[..., None] * np.array([1.0, 0.93, 0.8], np.float32)
    n = cv2.resize(C.fbm(pw // 8, ph // 8, 2.0, 3, seed=seed + 71), (pw, ph)) - 0.5
    sky = sky * (1 + 0.03 * n[..., None])

    def P(u, v):
        return (u - U0) * ppu, (V1 - v) * ppu

    specs = []
    # hero cumulus bank to the right of the view centre, a tower on the left, mid clouds
    for (u, v, wu, hu, d, k, fl) in [(1.5, 0.02, 0.62, 0.21, 2.0, 'cumulus', 1.0),
                                     (0.05, 0.05, 0.55, 0.30, 2.6, 'cumulus', 1.0),
                                     (0.62, 0.20, 0.32, 0.12, 3.2, 'cumulus', 0.9),
                                     (2.3, 0.10, 0.6, 0.24, 2.8, 'cumulus', 0.95), (1.08, 0.2, 0.16, 0.05, 3.6, 'cumulus', 0.85),
                                     (-0.75, 0.16, 0.4, 0.14, 3.4, 'stratocumulus', 0.85),
                                     (0.9, 0.42, 0.28, 0.08, 4.0, 'stratocumulus', 0.8),
                                     (1.9, 0.52, 0.34, 0.09, 4.2, 'stratocumulus', 0.8)]:
        x, y = P(u, v)
        specs.append((x, y, wu * ppu, hu * ppu, d, k, fl))
    frag = [s for s in S.cloud_specs(pw, ph, seed=seed + 11, kind='cumulus', coverage=0.45, horizon=hzp,
                                     fragments=1.3, strato=0.8)[1:] if (s[2] < 0.09 * pw and s[3] < 0.05 * pw)
            or s[5] == 'stratocumulus']
    kw = dict(seed=seed, sun_pos=sun_p, sun_z=0.3, palette=dict(
        lit_low='#fff1de', lit_high='#fffdf6', hi='#fffef8', mid='#c9bfe6', shd_high='#a3b0e6',
        shd_low='#6a78c8', bounce='#c8c0e4', base='#9496d2', rim='#fffbee', haze='#cfe8f8', fill='#8ec0f0'),
              sky=sky, rim=1.4, backlit=0.3, backlit_radius=0.22, lit_bias=0.12, detail=1.15)
    clouds = S.cumulus_plate(pw, ph, clouds=specs + frag, **kw)
    bank = S.horizon_bank(pw, ph, seed=seed + 3, palette='summer_noon', horizon=hzp, sun_pos=sun_p, sky=sky,
                          height=0.035, rows=3)
    cir = S.cirrus_plate(pw, ph, seed=seed + 8, color=(1.0, 0.98, 0.95), angle=-5, density=0.5,
                         region=(0.12, 0.45), opacity=0.45, clusters=3)
    img = sky
    for L in (cir, bank, clouds):
        img = C.over(img, L[..., :3], L[..., 3])
    # distant hazy town + hills sitting on the horizon (seen through gaps in the trees)
    img = _distant_town(img, ppu, hzp, seed)
    return img.astype(np.float32), sun_p


def _distant_town(img, ppu, hzp, seed):
    """Hazy distant town on the horizon: a tree-lined ridge, apartment blocks with sunlit faces and
    window rows, all melting into the horizon haze (atmospheric perspective)."""
    ph, pw = img.shape[:2]
    rng = np.random.default_rng(seed + 77)
    hy = hzp * ph
    haze = np.array([0.8, 0.9, 0.97], np.float32)
    Y = np.arange(ph, dtype=np.float32)[:, None]
    # far ridge with a bumpy tree-line top
    n = C.fbm(pw, 8, 5.0, 4, seed=seed + 5)[4]
    bumps = np.abs(np.sin(np.arange(pw) / (0.006 * ppu) + n * 6)) ** 0.5
    top = hy - (0.008 + 0.018 * n + 0.003 * bumps) * ppu
    m = C.smoothstep(-0.8, 0.8, Y - top[None, :])
    img = C.lerp(img, np.array([0.62, 0.76, 0.88], np.float32), (m * 0.55)[..., None])
    # buildings: two rows, nearer row slightly more contrast
    ss = 2
    for row, (hh, a) in enumerate([(0.016, 0.35), (0.024, 0.5)]):
        base = np.zeros((ph * ss, pw * ss), np.float32)
        lit = np.zeros_like(base)
        x = rng.uniform(0, 0.02) * ppu
        while x < pw:
            w = rng.uniform(0.012, 0.045) * ppu
            h = (0.3 + 0.7 * rng.random() ** 2) * hh * ppu
            if rng.random() < 0.35:
                x += w * rng.uniform(0.3, 1.5)
                continue
            y0 = hy - h + row * 0.002 * ppu
            X0, X1 = int(x * ss), int((x + w) * ss)
            Y0, Y1 = int(y0 * ss), int((hy + 0.01 * ppu) * ss)
            cv2.rectangle(base, (X0, Y0), (X1, Y1), 1.0, -1)
            # sun-facing (right) face sliver lit warm
            sw = max(1, int(w * rng.uniform(0.15, 0.35) * ss))
            cv2.rectangle(lit, (X1 - sw, Y0), (X1, Y1), 1.0, -1)
            # window rows
            for yy in np.arange(y0 + 0.002 * ppu, hy, 0.0035 * ppu):
                cv2.line(lit, (X0 + 2, int(yy * ss)), (X1 - sw - 2, int(yy * ss)), 0.35, 1)
            x += w + rng.uniform(0.0, 0.006) * ppu
        base = cv2.resize(base, (pw, ph), interpolation=cv2.INTER_AREA)
        lit = cv2.resize(lit, (pw, ph), interpolation=cv2.INTER_AREA)
        col = np.array([0.6, 0.72, 0.86], np.float32)
        lc = np.array([0.95, 0.9, 0.86], np.float32)
        c = C.lerp(np.broadcast_to(col, img.shape), lc, (lit * 0.6)[..., None])
        img = C.lerp(img, c, (base * a)[..., None])
    # distant hazy tree line in front of the town: rounded crowns, blue with atmospheric perspective,
    # a lighter sunlit top edge
    ss = 2
    tm = np.zeros((ph * ss, pw * ss), np.uint8)
    tl = np.zeros_like(tm)
    x = 0.0
    while x < pw:
        r = rng.uniform(0.004, 0.011) * ppu
        yb = hy + 0.004 * ppu - rng.uniform(0.0, 0.008) * ppu
        cv2.circle(tm, (int(x * ss), int(yb * ss)), int(r * ss), 255, -1, cv2.LINE_AA)
        cv2.circle(tl, (int((x + r * 0.25) * ss), int((yb - r * 0.3) * ss)), int(r * 0.8 * ss), 255, -1, cv2.LINE_AA)
        x += r * rng.uniform(0.7, 1.3)
    cv2.rectangle(tm, (0, int((hy + 0.004 * ppu) * ss)), (pw * ss, int((hy + 0.03 * ppu) * ss)), 255, -1)
    tm = cv2.resize(tm.astype(np.float32) / 255, (pw, ph), interpolation=cv2.INTER_AREA)
    tl = cv2.resize(tl.astype(np.float32) / 255, (pw, ph), interpolation=cv2.INTER_AREA) * tm
    tc = C.lerp(np.broadcast_to(np.array([0.36, 0.52, 0.62], np.float32), img.shape),
                np.array([0.58, 0.72, 0.74], np.float32), tl[..., None] * 0.8)
    img = C.lerp(img, tc, (tm * 0.85)[..., None])
    # haze band right on the horizon
    band = np.exp(-((Y - hy + 0.004 * ppu) / (0.012 * ppu)) ** 2)
    img = C.lerp(img, haze, (band * 0.45)[..., None])
    return img


# ----------------------------------------------------------------------------- trees

def _clump(canvas, depthc, cx, cy, rc, rng, ppm, L2, tone, ss=2, haze=0.0):
    """Paint one foliage clump into canvas in place, Shinkai style: a few big leaf-mass lobes with
    serrated (leafy) silhouettes, each posterised into hard-edged tones - a sunlit cap facing the light,
    a green mid band and a cool blue-teal underside; the clump's lower part falls into shadow."""
    h, w = canvas.shape[:2]
    R = int(rc * 1.55) + 4
    x0, y0 = int(cx) - R, int(cy) - R
    x1, y1 = int(cx) + R, int(cy) + R
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    pw_, ph_ = x1 - x0, y1 - y0
    PW, PH = pw_ * ss, ph_ * ss
    m = np.zeros((PH, PW), np.float32)
    T = np.zeros((PH, PW), np.float32)
    gx, gy = np.meshgrid(np.arange(PW, dtype=np.float32) / ss + x0, np.arange(PH, dtype=np.float32) / ss + y0)
    # leafy jitter field for the terminators (cell ~ 0.1 rc)
    cell = max(2, int(rc * 0.11))
    nz = rng.random((PH // (cell * ss) + 3, PW // (cell * ss) + 3)).astype(np.float32)
    nz = cv2.resize(nz, (PW + 2 * cell * ss, PH + 2 * cell * ss), interpolation=cv2.INTER_CUBIC)[:PH, :PW] - 0.5
    Lx, Ly = float(L2[0]), float(L2[1])
    nl = int(rng.integers(6, 11))
    lobes = [(cx, cy + rc * 0.1, rc * rng.uniform(0.5, 0.6))]
    for i in range(nl):
        a = rng.uniform(0, 2 * math.pi)
        r = rc * rng.uniform(0.3, 0.62)
        lobes.append((cx + math.cos(a) * r * 1.05, cy + math.sin(a) * r * 0.8, rc * rng.uniform(0.28, 0.46)))
    lobes.sort(key=lambda q: -q[1])                     # lowest first, upper lobes overlap them
    for (lx, ly, lr) in lobes:
        # work only inside the lobe's bounding box
        bx0 = max(0, int((lx - lr * 1.3 - x0) * ss))
        by0 = max(0, int((ly - lr * 1.3 - y0) * ss))
        bx1 = min(PW, int((lx + lr * 1.3 - x0) * ss) + 2)
        by1 = min(PH, int((ly + lr * 1.3 - y0) * ss) + 2)
        if bx1 <= bx0 or by1 <= by0:
            continue
        cm = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        c0 = (int((lx - x0) * ss) - bx0, int((ly - y0) * ss) - by0)
        cv2.circle(cm, c0, max(1, int(lr * ss)), 255, -1, cv2.LINE_AA)
        # serrated rim: small leaf tufts around the upper / sun side of the lobe
        for k in range(int(16 + lr / 2)):
            a = rng.uniform(-math.pi, math.pi)
            up = -math.sin(a)
            if up < -0.4 and rng.random() < 0.75:
                continue
            rr = lr * rng.uniform(0.08, 0.19)
            d = lr * rng.uniform(0.88, 1.02)
            p = (int((lx + math.cos(a) * d - x0) * ss) - bx0, int((ly + math.sin(a) * d - y0) * ss) - by0)
            cv2.circle(cm, p, max(1, int(rr * ss)), 255, -1, cv2.LINE_AA)
        a = cm.astype(np.float32) / 255.0
        gxs, gys = gx[by0:by1, bx0:bx1], gy[by0:by1, bx0:bx1]
        dx, dy = (gxs - lx) / lr, (gys - ly) / lr
        sh = dx * Lx + dy * Ly + np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1)) * 0.35
        sh = sh - 0.55 * np.clip((gys - cy) / rc, -1, 1) + nz[by0:by1, bx0:bx1] * 0.5 + tone * 0.25
        v = 0.12 + 0.3 * C.smoothstep(-0.32, -0.26, sh) + 0.32 * C.smoothstep(0.2, 0.26, sh)             + 0.2 * C.smoothstep(0.62, 0.68, sh)
        Tb = T[by0:by1, bx0:bx1]
        T[by0:by1, bx0:bx1] = Tb * (1 - a) + v * a
        m[by0:by1, bx0:bx1] = np.maximum(m[by0:by1, bx0:bx1], a)
    mf = cv2.resize(m, (pw_, ph_), interpolation=cv2.INTER_AREA)
    Tf = cv2.resize(T * m, (pw_, ph_), interpolation=cv2.INTER_AREA) / np.maximum(mf, 1e-3)
    stops = np.array([0.0, 0.12, 0.42, 0.74, 0.94], np.float32)
    cols = np.array([[0.02, 0.07, 0.11], [0.05, 0.17, 0.22], [0.12, 0.36, 0.2], [0.44, 0.66, 0.2],
                     [0.86, 0.95, 0.52]], np.float32)
    Tc = np.clip(Tf, 0, 1)
    col = np.stack([np.interp(Tc, stops, cols[:, c]) for c in range(3)], -1).astype(np.float32)
    if haze > 0:
        col = C.lerp(col, np.array([0.55, 0.72, 0.82], np.float32), haze)
    sub = canvas[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    cy0, cx0 = max(y0, 0) - y0, max(x0, 0) - x0
    sl = (slice(cy0, cy0 + sub.shape[0]), slice(cx0, cx0 + sub.shape[1]))
    # soft cast shadow of this clump onto what is already painted below / behind it
    M3 = np.float32([[1, 0, -L2[0] * rc * 0.3], [0, 1, -L2[1] * rc * 0.3]])
    shad = C.blur(cv2.warpAffine(mf, M3, (pw_, ph_), borderValue=0), rc * 0.12)
    sub[..., :3] *= (1 - 0.5 * shad[sl])[..., None]
    a = mf[sl][..., None]
    sub[..., :3] = sub[..., :3] * (1 - a) + col[sl] * a
    sub[..., 3:4] = np.maximum(sub[..., 3:4], a)


_LEAF_STOPS = np.array([0.0, 0.14, 0.36, 0.62, 0.86, 1.0], np.float32)
_LEAF_COLS = np.array([[0.02, 0.07, 0.10], [0.05, 0.16, 0.20], [0.11, 0.33, 0.24], [0.33, 0.58, 0.22],
                       [0.66, 0.82, 0.30], [0.92, 0.97, 0.62]], np.float32)


def _paint_tree(canvas, clumps, rng, L2, haze, cyc, ry, ss=2):
    """Paint one tree crown from leaf dabs (Shinkai foliage): every clump = a dark base mass + a few
    hundred small rotated leaf dabs posterised into 4 tones by a sphere-ish light term (lit dabs drawn
    last so the sunlit leaves overlap the shade crisply), leafy broken silhouettes, a few sky holes."""
    h, w = canvas.shape[:2]
    pad = 8
    x0 = int(min(c[0] - c[2] * 1.2 for c in clumps)) - pad
    x1 = int(max(c[0] + c[2] * 1.2 for c in clumps)) + pad
    y0 = int(min(c[1] - c[2] * 1.2 for c in clumps)) - pad
    y1 = int(max(c[1] + c[2] * 1.2 for c in clumps)) + pad
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return
    PW, PH = (x1 - x0) * ss, (y1 - y0) * ss
    tone = np.zeros((PH, PW), np.uint8)
    m = np.zeros((PH, PW), np.uint8)
    Lx, Ly = float(L2[0]), float(L2[1])

    def P(x, y):
        return int((x - x0) * ss), int((y - y0) * ss)

    for (cx, cy, rc, tb) in clumps:
        # base mass: a few overlapping discs, dark
        for k in range(7):
            a = rng.uniform(0, 2 * math.pi)
            r = rc * rng.uniform(0.0, 0.2)
            rr = rc * rng.uniform(0.42, 0.56)
            px, py = cx + math.cos(a) * r, cy + math.sin(a) * r * 0.8
            dx, dy = (px - cx) / rc, (py - cy) / rc
            t0 = 0.22 + 0.12 * max(0.0, dx * Lx + dy * Ly) - 0.1 * max(0.0, (py - cyc) / ry)
            cv2.circle(tone, P(px, py), int(rr * ss), int(255 * t0), -1)
            cv2.circle(m, P(px, py), int(rr * ss), 255, -1)
        n = int(np.clip(rc * rc / 12.0, 80, 1600))
        ang = rng.uniform(0, 2 * math.pi, n)
        # irregular (non-circular) clump outline + a few leafy sprigs poking out of it
        hk = rng.uniform(0, 2 * math.pi, 4)
        edge = (1.0 + 0.16 * np.sin(2 * ang + hk[0]) + 0.12 * np.sin(3 * ang + hk[1])
                + 0.08 * np.sin(5 * ang + hk[2]) + 0.05 * np.sin(9 * ang + hk[3]))
        rad = rng.uniform(0, 1, n) ** 0.5 * 1.02 * edge
        nsp = int(rng.integers(3, 8))
        for k in range(nsp):
            a0 = rng.uniform(0, 2 * math.pi)
            sel = rng.random(n) < 0.012
            ang[sel] = a0 + rng.normal(0, 0.12, sel.sum())
            rad[sel] = rng.uniform(0.9, 1.3, sel.sum())
        dx = np.cos(ang) * rad
        dy = np.sin(ang) * rad * 0.85
        zz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
        sh = (dx * Lx + dy * Ly) * 0.95 + zz * 0.2 - 0.5 * np.clip((cy + dy * rc - cyc) / ry, -1, 1)             + rng.normal(0, 0.075, n) + tb - 0.22
        lev = np.where(sh < -0.1, 0.18, np.where(sh < 0.25, 0.4, np.where(sh < 0.55, 0.66, 0.9)))
        lev = lev + rng.normal(0, 0.035, n)
        size = rc * rng.uniform(0.045, 0.085, n) * (1.1 - 0.35 * (lev > 0.8))
        o = np.argsort(lev + rng.uniform(0, 0.05, n))
        for i in o:
            px, py = cx + dx[i] * rc, cy + dy[i] * rc
            a_ = math.degrees(ang[i]) + rng.normal(0, 40)
            ax_ = (max(1, int(size[i] * ss)), max(1, int(size[i] * ss * rng.uniform(0.42, 0.62))))
            c = P(px, py)
            cv2.ellipse(tone, c, ax_, a_, 0, 360, int(255 * np.clip(lev[i], 0.02, 1.0)), -1)
            cv2.ellipse(m, c, ax_, a_, 0, 360, 255, -1)
        # sky holes near the rim of the clump
        for k in range(int(rng.integers(2, 7))):
            a = rng.uniform(0, 2 * math.pi)
            r = rc * rng.uniform(0.62, 0.9)
            rr = rc * rng.uniform(0.03, 0.06)
            c = P(cx + math.cos(a) * r, cy + math.sin(a) * r * 0.85)
            cv2.ellipse(m, c, (max(1, int(rr * ss)), max(1, int(rr * ss * 0.6))), rng.uniform(0, 180), 0, 360, 0, -1)
    mf = cv2.resize(m.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    tf = cv2.resize(tone.astype(np.float32) / 255.0 * (m > 0), (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_AREA) / np.maximum(mf, 1e-3)
    col = np.stack([np.interp(np.clip(tf, 0, 1), _LEAF_STOPS, _LEAF_COLS[:, c]) for c in range(3)], -1)
    col = col.astype(np.float32)
    if haze > 0:
        col = C.lerp(col, np.array([0.55, 0.72, 0.82], np.float32), haze * (0.6 + 0.4 * (1 - tf))[..., None])
    sl = (slice(y0c - y0, y1c - y0), slice(x0c - x0, x1c - x0))
    sub = canvas[y0c:y1c, x0c:x1c]
    a = mf[sl][..., None]
    # soft contact shadow of this crown on what is behind it
    shd = C.blur(mf, 6.0)[sl][..., None]
    sub[..., :3] *= 1 - 0.35 * shd * (1 - a)
    sub[..., :3] = sub[..., :3] * (1 - a) + col[sl] * a
    sub[..., 3:4] = np.maximum(sub[..., 3:4], a)


_CF_STOPS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], np.float32)
_CF_COLS = np.array([[0.025, 0.085, 0.115],     # deep cool core
                     [0.055, 0.19, 0.21],       # blue-green shadow mass
                     [0.13, 0.35, 0.22],        # mid green
                     [0.36, 0.6, 0.18],         # sunlit leaf green
                     [0.68, 0.84, 0.26],        # warm yellow-green (backlit, translucent)
                     [0.93, 0.98, 0.6]], np.float32)   # hot rim


def _leaf_cluster(lab, cx, cy, R, k, rng, tips=(5, 9)):
    """Fill an irregular leaf-cluster shape (pointed leaf tips with notches between them, slightly
    drooping, randomly rotated) into the label image with value k."""
    if R < 2.0:
        cv2.circle(lab, (int(cx), int(cy)), max(1, int(R)), int(k), -1)
        return
    m = int(rng.integers(tips[0], tips[1]))
    a0 = rng.uniform(0, 2 * math.pi)
    ang = a0 + (np.arange(2 * m) + rng.uniform(-0.3, 0.3, 2 * m)) * (math.pi / m)
    rr = np.where(np.arange(2 * m) % 2 == 0, rng.uniform(0.92, 1.15, 2 * m), rng.uniform(0.64, 0.82, 2 * m)) * R
    xs = cx + np.cos(ang) * rr
    ys = cy + np.sin(ang) * rr * 0.86 + np.maximum(np.sin(ang), 0) * rr * 0.12
    pts = np.stack([xs * 8, ys * 8], 1).astype(np.int32)
    cv2.fillPoly(lab, [pts], int(k), shift=3)


def _paint_crown(canvas, clumps, rng, L2, haze, cyc, ry, ss=2):
    """Cauliflower crown (Shinkai foliage). Every clump is built from leaf-cluster discs (big inside,
    small at the rim) painted back-to-front into a label image, so silhouettes and the tone terminators
    are hard scalloped arcs. Each disc is shaded as its own little dome (lit cap toward the sun, cool
    underside) on top of the clump's big form and the crown's vertical falloff, then posterised into
    flat painted tones. A crisp rim of the hottest tone runs along the sun-side silhouette; a few sky
    holes are punched through."""
    h, w = canvas.shape[:2]
    pad = 8
    x0 = int(min(c[0] - c[2] * 1.2 for c in clumps)) - pad
    x1 = int(max(c[0] + c[2] * 1.2 for c in clumps)) + pad
    y0 = int(min(c[1] - c[2] * 1.2 for c in clumps)) - pad
    y1 = int(max(c[1] + c[2] * 1.2 for c in clumps)) + pad
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c <= x0c or y1c <= y0c:
        return
    PW, PH = (x1 - x0) * ss, (y1 - y0) * ss
    lab = np.full((PH, PW), -1, np.int32)
    Lx, Ly = float(L2[0]), float(L2[1])
    DX, DY, DR, DB = [], [], [], []
    for (cx, cy, rc, tb) in clumps:
        n = int(np.clip(60 + rc * 0.9, 70, 220))
        a = rng.uniform(0, 2 * math.pi, n)
        rr = np.sqrt(rng.uniform(0, 1, n)) * 0.86
        # irregular clump outline
        hk = rng.uniform(0, 2 * math.pi, 3)
        edge = 1.0 + 0.12 * np.sin(2 * a + hk[0]) + 0.08 * np.sin(3 * a + hk[1]) + 0.05 * np.sin(5 * a + hk[2])
        dx = np.cos(a) * rr * edge
        dy = np.sin(a) * rr * edge * 0.82
        rad = rc * (0.06 + 0.13 * (1 - rr) ** 1.2) * rng.uniform(0.75, 1.25, n)
        px, py = cx + dx * rc, cy + dy * rc
        zz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
        big = ((dx * Lx + dy * Ly) * 0.7 + zz * 0.2 - 0.6 * np.clip((py - cyc) / ry, -1, 1) + tb - 0.34
               + rng.normal(0, 0.03, n))
        # paint order: lower / shadow-side discs first, upper sunlit ones on top
        o = np.argsort((py - cy) / rc * -1.0 + (dx * Lx + dy * Ly) * 0.3 + rng.normal(0, 0.2, n))
        for i in o:
            k = len(DX)
            DX.append(px[i]); DY.append(py[i]); DR.append(rad[i]); DB.append(big[i])
            _leaf_cluster(lab, (px[i] - x0) * ss, (py[i] - y0) * ss, rad[i] * ss, k, rng)
        # a few tiny leafy scallops on the sun side of the rim (fine cauliflower edge, stays attached)
        m = int(rng.integers(10, 22))
        a = rng.normal(math.atan2(Ly, Lx), 0.9, m)
        rr = rng.uniform(0.74, 0.86, m)
        px2, py2 = cx + np.cos(a) * rr * rc, cy + np.sin(a) * rr * rc * 0.82
        for i in range(m):
            k = len(DX)
            DX.append(px2[i]); DY.append(py2[i]); DR.append(rc * rng.uniform(0.045, 0.07))
            DB.append(0.4 - 0.6 * float(np.clip((py2[i] - cyc) / ry, -1, 1)) + tb)
            _leaf_cluster(lab, (px2[i] - x0) * ss, (py2[i] - y0) * ss, DR[-1] * ss, k, rng, tips=(4, 7))
        # sky holes inside the outer half of the clump
        for k in range(int(rng.integers(2, 7))):
            a_ = rng.uniform(0, 2 * math.pi)
            r_ = rc * rng.uniform(0.45, 0.8)
            hr = rc * rng.uniform(0.03, 0.08)
            c = (int((cx + math.cos(a_) * r_ - x0) * ss), int((cy + math.sin(a_) * r_ * 0.8 - y0) * ss))
            cv2.ellipse(lab, c, (max(1, int(hr * ss)), max(1, int(hr * ss * rng.uniform(0.5, 0.9)))),
                        rng.uniform(0, 180), 0, 360, -1, -1)
    DX, DY, DR, DB = (np.array(v, np.float32) for v in (DX, DY, DR, DB))
    msk = lab >= 0
    li = np.where(msk, lab, 0)
    gx = (np.arange(PW, dtype=np.float32) / ss + x0)[None, :]
    gy = (np.arange(PH, dtype=np.float32) / ss + y0)[:, None]
    ux = (gx - DX[li]) / DR[li]
    uy = (gy - DY[li]) / DR[li]
    uz = np.sqrt(np.clip(1 - ux * ux - uy * uy, 0, 1))
    small = (ux * Lx + uy * Ly) * 0.11 + uz * 0.03
    sh = DB[li] + small
    # posterise into flat tones with a hair of anti-aliasing at the terminators
    lev = (0.08 + 0.2 * C.smoothstep(-0.52, -0.46, sh) + 0.2 * C.smoothstep(-0.12, -0.06, sh)
           + 0.2 * C.smoothstep(0.24, 0.3, sh) + 0.18 * C.smoothstep(0.58, 0.64, sh))
    # crisp hot rim on the sun-side silhouette (2 plate px)
    sft = int(max(2, 2 * ss))
    M = np.float32([[1, 0, -Lx * sft], [0, 1, -Ly * sft]])
    beyond = cv2.warpAffine(msk.astype(np.uint8), M, (PW, PH), flags=cv2.INTER_NEAREST, borderValue=0)
    rim = msk & (beyond == 0) & (sh > -0.25)
    lev = np.where(rim, np.maximum(lev, 0.97), lev)
    lev = np.where(msk, lev, 0.0).astype(np.float32)
    mf = cv2.resize(msk.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    tf = cv2.resize(lev, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) / np.maximum(mf, 1e-3)
    col = np.stack([np.interp(np.clip(tf, 0, 1), _CF_STOPS, _CF_COLS[:, c]) for c in range(3)], -1)
    col = col.astype(np.float32)
    if haze > 0:
        col = C.lerp(col, np.array([0.55, 0.72, 0.82], np.float32), haze * (0.6 + 0.4 * (1 - tf))[..., None])
    sl = (slice(y0c - y0, y1c - y0), slice(x0c - x0, x1c - x0))
    sub = canvas[y0c:y1c, x0c:x1c]
    a = mf[sl][..., None]
    shd = C.blur(mf, 6.0)[sl][..., None]
    sub[..., :3] *= 1 - 0.35 * shd * (1 - a)
    sub[..., :3] = sub[..., :3] * (1 - a) + col[sl] * a
    sub[..., 3:4] = np.maximum(sub[..., 3:4], a)


def tree_plate(ppm, sun_dir_screen=(0.25, -1.0), seed=9):
    """RGBA plate of backlit tree crowns (+ trunks/branches) on the plane z = TREE_Z."""
    pw, ph = int((TX1 - TX0) * ppm), int((TY1 - TY0) * ppm)
    canvas = np.zeros((ph, pw, 4), np.float32)
    rng = np.random.default_rng(seed)
    L2 = np.array(sun_dir_screen, np.float32)
    L2 /= np.linalg.norm(L2)

    def P(x, y):
        return (x - TX0) * ppm, (TY1 - y) * ppm

    trees = []
    x = TX0 + 1.0
    while x < TX1 - 1:
        top = rng.uniform(0.8, 4.2)
        rx = rng.uniform(2.6, 4.2)
        trees.append((x, top, rx))
        x += rx * rng.uniform(1.1, 1.6)
    # distant tree line: small, hazy blue-green crowns low behind everything (aerial perspective)
    frng = np.random.default_rng(seed + 100)
    x = TX0 + 0.5
    while x < TX1 - 0.5:
        rxf = frng.uniform(0.9, 1.8)
        topf = frng.uniform(1.0, 2.1)
        ryf = rxf * frng.uniform(0.7, 0.95)
        clumps = []
        for i in range(int(6 + rxf * 3)):
            ang = frng.uniform(0, 2 * math.pi)
            r = math.sqrt(frng.uniform(0, 1))
            ccx = x + math.cos(ang) * r * rxf * 0.8
            ccy = topf - ryf + math.sin(ang) * r * ryf * 0.75
            rc = rxf * frng.uniform(0.28, 0.42)
            px_, py_ = P(ccx, ccy)
            clumps.append((px_, py_, rc * ppm, frng.uniform(-0.1, 0.15)))
        clumps.sort(key=lambda c: -c[1])
        _paint_crown(canvas, clumps, frng, L2, 0.62, P(0, topf - ryf)[1], ryf * ppm)
        x += rxf * frng.uniform(0.9, 1.4)
    # far trees first
    order = sorted(range(len(trees)), key=lambda i: trees[i][1] < 1.8)
    ground = -6.0
    for ti in order:
        tx, top, rx = trees[ti]
        ry = rx * rng.uniform(0.8, 1.05)
        cyc = top - ry
        bx, by = P(tx + rng.uniform(-0.3, 0.3), ground)
        cxp, cyp = P(tx, cyc)
        trunk = np.array([0.16, 0.14, 0.15], np.float32)
        wpx = max(2, int(0.24 * ppm))
        m = np.zeros((ph, pw), np.uint8)
        cv2.line(m, (int(bx), int(by)), (int(cxp), int(cyp + 0.2 * ry * ppm)), 255, wpx, cv2.LINE_AA)
        for b in range(14):
            ang = rng.uniform(-2.4, -0.7)
            ln = rng.uniform(0.35, 0.75) * rx * ppm
            sx, sy = cxp + rng.uniform(-0.2, 0.2) * rx * ppm, cyp + rng.uniform(0.0, 0.5) * ry * ppm
            ex, ey = sx + math.cos(ang) * ln, sy + math.sin(ang) * ln
            # keep the branch inside the crown (it only shows through the gaps)
            q = math.hypot((ex - cxp) / (0.62 * rx * ppm), (ey - cyp) / (0.6 * ry * ppm))
            if q > 1.0:
                ex, ey = cxp + (ex - cxp) / q, cyp + (ey - cyp) / q
            cv2.line(m, (int(sx), int(sy)), (int(ex), int(ey)), 255, max(1, wpx // 3), cv2.LINE_AA)
            for tw_ in range(2):
                a2 = ang + rng.uniform(-0.8, 0.8)
                l2 = ln * rng.uniform(0.12, 0.25)
                tx_, ty_ = ex + math.cos(a2) * l2, ey + math.sin(a2) * l2
                q = math.hypot((tx_ - cxp) / (0.66 * rx * ppm), (ty_ - cyp) / (0.64 * ry * ppm))
                if q > 1.0:
                    tx_, ty_ = cxp + (tx_ - cxp) / q, cyp + (ty_ - cyp) / q
                cv2.line(m, (int(ex), int(ey)), (int(tx_), int(ty_)), 255,
                         max(1, wpx // 6), cv2.LINE_AA)
        a = m.astype(np.float32) / 255
        canvas[..., :3] = canvas[..., :3] * (1 - a[..., None]) + trunk * a[..., None]
        canvas[..., 3] = np.maximum(canvas[..., 3], a)
        n = int(12 + rx * 4)
        thz = rng.uniform(0.0, 0.1) + 0.16 * (top < 1.8)
        cl = []
        for i in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            r = math.sqrt(rng.uniform(0, 1)) ** 0.6
            ccx = tx + math.cos(ang) * r * rx * 0.82
            ccy = cyc + math.sin(ang) * r * ry * 0.78
            rc = rx * rng.uniform(0.22, 0.36) * (1.15 - 0.3 * r)
            depth = rng.uniform(0, 1) + (1 - r) * 0.4 - (ccy - cyc) / ry * 0.2
            cl.append((depth, ccx, ccy, rc))
        cl.sort()
        clumps = []
        for depth, ccx, ccy, rc in cl:
            px, py = P(ccx, ccy)
            clumps.append((px, py, rc * ppm, depth * 0.3 - 0.15))
        _paint_crown(canvas, clumps, rng, L2, thz, P(0, cyc)[1], ry * ppm)
    # the lower part fades into dense dark foliage (the trees continue to the ground)
    ys = np.arange(ph, dtype=np.float32)
    yw = TY1 - ys / ppm
    fill = C.smoothstep(-1.0, -3.5, yw)[:, None]
    base = np.array([0.03, 0.09, 0.1], np.float32)
    canvas[..., :3] = C.lerp(canvas[..., :3], base, (fill * (1 - canvas[..., 3:4][..., 0]))[..., None] * 1.0)
    canvas[..., 3] = np.maximum(canvas[..., 3], fill[:, 0][:, None] * np.ones((1, pw), np.float32))
    return canvas.astype(np.float32)
