"""Painted snow-laden conifer for s08_snow_station.

Each tree is built from irregular drooping branch tiers; every arm is a chain of lumpy needle pads
(ragged hanging needle tufts underneath) carrying pillowy snow caps that are shaded from a bright
cold-white top to a blue-violet underside, with a thin dark crevice where the snow meets the needles.
Extra pads scattered across the crown face break the symmetry. A cool sky rim is added on the
sky-facing silhouette and a warm tint on the side facing the station lamps."""
import math
import numpy as np
import cv2


def _blob(cx, cy, a, b, rng, rot=0.0, n=40, rough=0.08, tuft=0.0, tuft_n=9, flat_top=0.0):
    """Irregular closed blob outline around (cx, cy), half-sizes a, b. The lower half can carry ragged
    hanging needle tufts (tuft = relative length)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ph = rng.uniform(0, 6.28, 3)
    r = 1 + rough * (np.sin(th * 3 + ph[0]) * 0.6 + np.sin(th * 5 + ph[1]) * 0.4) + rng.normal(0, rough * 0.35, n)
    y = np.sin(th)                      # >0 = lower half (screen y down)
    if tuft > 0:
        k = np.abs(np.sin(th * tuft_n * 0.5 + ph[2]))
        jag = rng.uniform(0.2, 1.0, n) * k ** 1.5
        r = r + tuft * jag * np.clip(y, 0, 1) ** 0.7
    if flat_top > 0:
        r = r * (1 - flat_top * np.clip(-y, 0, 1) ** 2)
    xs = a * r * np.cos(th)
    ys = b * r * np.sin(th)
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + xs * c - ys * s, cy + xs * s + ys * c], 1)


def pine(cv, bu, bv, hpx, z, seed, fol=(0.035, 0.07, 0.11), snow_top=(0.66, 0.76, 0.98),
         snow_shd=(0.2, 0.26, 0.5), rim=(0.72, 0.84, 1.1), warm=None, warm_side=0.0, warm_amt=0.0,
         fog=None, fog_amt=0.0, width=0.27, snow=1.0, ss=2, rim_amt=1.0, lean=None, detail=1.0):
    """Paint a snow-laden conifer standing at screen (bu, bv) with pixel height hpx into canvas cv."""
    rng = np.random.default_rng(seed)
    if hpx < 3:
        return
    rim_dir = float(np.sign(warm_side)) if (warm is not None and warm_amt > 0.05) else 0.0
    if hpx < 40:
        ss = max(ss, 3)
    hw0 = hpx * width * rng.uniform(0.85, 1.15)
    x0 = int(bu - hw0 * 1.6) - 4
    y0 = int(bv - hpx * 1.06) - 4
    w = int(hw0 * 3.2) + 8
    h = int(hpx * 1.1) + 8
    S = ss
    Hh, Ww = h * S, w * S
    col = np.zeros((Hh, Ww, 3), np.float32)
    alp = np.zeros((Hh, Ww), np.float32)
    snowm = np.zeros((Hh, Ww), np.float32)
    cx = (bu - x0) * S
    top = (bv - hpx - y0) * S
    H = hpx * S
    hw = hw0 * S
    fol = np.asarray(fol, np.float32)
    s_top = np.asarray(snow_top, np.float32)
    s_shd = np.asarray(snow_shd, np.float32)
    rimc = np.asarray(rim, np.float32)

    def fill(poly, c0, c1=None, ya=0.0, yb=1.0, sn=0.0):
        """Fill polygon with colour c0 (or a vertical gradient c0 at y=ya -> c1 at y=yb)."""
        p = np.round(poly).astype(np.int32)
        bx0, by0 = max(p[:, 0].min(), 0), max(p[:, 1].min(), 0)
        bx1, by1 = min(p[:, 0].max() + 1, Ww), min(p[:, 1].max() + 1, Hh)
        if bx1 <= bx0 or by1 <= by0:
            return
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillPoly(m, [p - [bx0, by0]], 1)
        mb = m.astype(bool)
        if not mb.any():
            return
        if c1 is None:
            col[by0:by1, bx0:bx1][mb] = c0
        else:
            yy = np.arange(by0, by1, dtype=np.float32)[:, None]
            tt = np.clip((yy - ya) / max(yb - ya, 1e-3), 0, 1)
            g = c0[None, None, :] * (1 - tt[..., None]) + c1[None, None, :] * tt[..., None]
            g = np.broadcast_to(g, (by1 - by0, bx1 - bx0, 3))
            col[by0:by1, bx0:bx1][mb] = g[mb]
        alp[by0:by1, bx0:bx1][mb] = 1.0
        snowm[by0:by1, bx0:bx1][mb] = sn

    ph1, ph2 = rng.uniform(0, 6.28, 2)

    def env(r):
        return hw * (0.04 + 0.96 * r ** 0.92) * (1 + 0.1 * math.sin(r * 9 + ph1) + 0.06 * math.sin(r * 21 + ph2))

    # trunk
    tw = max(hpx * 0.018, 0.6) * S
    fill(np.array([[cx - tw * 0.6, top + H * 0.2], [cx + tw * 0.6, top + H * 0.2], [cx + tw * 1.4, top + H],
                   [cx - tw * 1.4, top + H]]), fol * 0.6)
    lean = rng.uniform(-0.025, 0.025) if lean is None else lean

    def pad(px, py, a, b, rot, shade, snow_amt, outer_dir):
        """One needle pad with its snow pillow."""
        fr = min(max(b / (S * 2.0), 0), 1)       # tufts only when big enough to read
        poly = _blob(px, py, a, b, rng, rot, n=36 if a > 6 * S else 20, rough=0.07,
                     tuft=0.85 * fr, tuft_n=int(rng.integers(5, 11)), flat_top=0.15)
        fc = fol * shade
        fill(poly, fc * 1.6, fc * 0.75, py - b, py + b * 1.3)
        if snow_amt <= 0.05:
            return
        # snow pillow: an irregular cap over the upper part of the pad, overhanging the outer end
        sa = a * rng.uniform(0.8, 1.05)
        sb = b * snow_amt * rng.uniform(0.6, 1.0)
        scx = px + outer_dir * a * rng.uniform(0.0, 0.18)
        scy = py - b * rng.uniform(0.25, 0.5)
        cap = _blob(scx, scy, sa, sb, rng, rot, n=40 if sa > 6 * S else 20, rough=0.1, flat_top=-0.3)
        # clip the cap to the upper half-ish: flatten its underside into a gentle sag
        cap[:, 1] = np.minimum(cap[:, 1], scy + sb * 0.55 + (cap[:, 0] - scx) * math.tan(rot))
        # thin dark crevice under the pillow
        crev = cap.copy()
        crev[:, 1] += max(sb * 0.14, S * 0.6)
        fill(crev, fol * 0.45, sn=0.0)
        sh = rng.uniform(0.9, 1.06)
        # painted cap: a crisp bright rim along the top edge (thicker on the lamp side), then the body
        # shaded from lit top to a cool blue-violet underside, then a darker shadow lip on the base
        rw = max(sb * 0.16, S * 0.9)
        fill(cap, rimc * 1.08, sn=2.0)
        body = cap + np.array([-rim_dir * rw * 0.6, rw])
        body[:, 1] = np.maximum(body[:, 1], cap[:, 1])
        fill(body, s_top * sh * 1.0, s_shd * sh, scy - sb * 0.7, scy + sb * 0.6, sn=1.0)
        lip = cap.copy()
        lip[:, 1] = np.maximum(lip[:, 1], scy + sb * 0.18)
        fill(lip, s_shd * sh * 0.82, sn=1.0)
        # lumps on the pillow top (2-3 bulges, brighter)
        nl = int(rng.integers(1, 4)) if sa > 4 * S else 0
        for _ in range(nl):
            lx = scx + rng.uniform(-0.6, 0.6) * sa
            la = sa * rng.uniform(0.25, 0.45)
            lb = sb * rng.uniform(0.45, 0.7)
            ly = scy - sb * rng.uniform(0.35, 0.6)
            lp = _blob(lx, ly, la, lb, rng, rot, n=18, rough=0.08)
            lp[:, 1] = np.minimum(lp[:, 1], ly + lb * 0.35)
            fill(lp, rimc * 1.02, sn=2.0)
            lp2 = lp + np.array([-rim_dir * rw * 0.4, rw * 0.7])
            lp2[:, 1] = np.maximum(lp2[:, 1], lp[:, 1])
            fill(lp2, s_top * sh * 1.06, s_top * sh * 0.8, ly - lb, ly + lb * 0.4, sn=1.0)

    n = int(np.clip(6 + hpx / 34, 6, 22) * rng.uniform(0.9, 1.12))
    rk = np.cumsum(rng.uniform(0.55, 1.5, n))
    rk = 0.07 + 0.87 * (rk - rk[0]) / (rk[-1] - rk[0])
    for k in range(n - 1, -1, -1):
        r = rk[k]
        e = env(r)
        th = max(H * 0.9 / n * rng.uniform(0.75, 1.2), S * 1.5)
        y = top + r * H
        xc_ = cx + lean * H * (1 - r)
        tier_snow = snow * (1.0 if rng.random() > 0.1 else 0.45)
        # back clumps around the trunk (fill the tier's core)
        pad(xc_, y + th * 0.25, e * 0.45, th * 0.55, 0.0, rng.uniform(0.8, 0.95), tier_snow * 0.9, 0)
        for side in (-1, 1):
            L = e * rng.uniform(0.72, 1.12)
            if rng.random() < 0.12:
                L *= rng.uniform(0.45, 0.7)
            droop = th * rng.uniform(0.8, 2.0)
            m = int(np.clip(round(L / max(th * 1.2, 1)), 1, 4 if detail >= 1 else 2))
            for j in range(m):
                d = L * (j + 0.75) / (m + 0.25)
                a = L / (m + 0.25) * rng.uniform(0.65, 0.95)
                if j == m - 1:
                    a *= rng.uniform(0.8, 1.05)
                b = th * rng.uniform(0.38, 0.62) * (1 - 0.25 * j / max(m, 1))
                slope = droop * 1.5 * (d / max(L, 1)) ** 0.5 / max(L, 1)
                rot = side * math.atan(slope) * 0.8
                py = y + droop * (d / max(L, 1)) ** 1.5
                pad(xc_ + side * d, py, a, b, rot, rng.uniform(0.85, 1.15), tier_snow * rng.uniform(0.8, 1.1), side)
        # scattered front pads on the crown face (break the silhouette-only look)
        if r > 0.18:
            for _ in range(int(rng.integers(0, 3 if detail >= 1 else 2))):
                fx = xc_ + rng.uniform(-0.55, 0.55) * e
                pad(fx, y + th * rng.uniform(0.35, 0.8), e * rng.uniform(0.18, 0.32), th * rng.uniform(0.3, 0.5),
                    rng.uniform(-0.15, 0.15), rng.uniform(0.95, 1.2), snow * rng.uniform(0.7, 1.1), np.sign(fx - xc_))
    # spire
    tipw = max(hw * 0.035, S)
    fill(np.array([[cx - tipw, top + H * 0.08], [cx + lean * H * 0.1, top - H * 0.02], [cx + tipw, top + H * 0.08]]), fol)
    cap = _blob(cx, top + H * 0.045, tipw * 1.5, H * 0.022, rng, 0, n=16, rough=0.1)
    fill(cap, s_top * 1.08, s_shd, top + H * 0.02, top + H * 0.07, sn=1.0)

    rimm = (snowm > 1.5).astype(np.float32)
    snowm = np.minimum(snowm, 1.0)
    # painterly brush texture inside the snow: soft horizontal strokes (+-6%)
    bt = rng.random((max(Hh // (5 * S), 2), max(Ww // (14 * S), 2))).astype(np.float32)
    bt = cv2.resize(bt, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    bt = cv2.GaussianBlur(bt, (0, 0), sigmaX=3.0 * S, sigmaY=0.8 * S)
    col = col * (1 + (snowm * (1 - rimm))[..., None] * (bt[..., None] - 0.5) * 0.14)
    # needle texture on the dark foliage: small streaky clusters
    nt = rng.random((max(Hh // (3 * S), 2), max(Ww // (3 * S), 2))).astype(np.float32)
    nt = cv2.resize(nt, (Ww, Hh), interpolation=cv2.INTER_LINEAR)
    nt = cv2.GaussianBlur(nt, (0, 0), sigmaX=0.6 * S, sigmaY=1.4 * S)
    fm = (1 - snowm)[..., None]
    col = col * (1 + fm * (nt[..., None] - 0.5) * 0.7)

    # downsample
    colL = cv2.resize(col * alp[..., None], (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alp, (w, h), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(snowm * alp, (w, h), interpolation=cv2.INTER_AREA)
    rm = cv2.resize(rimm * alp, (w, h), interpolation=cv2.INTER_AREA)
    colL = colL / np.maximum(a, 1e-4)[..., None]
    sm = sm / np.maximum(a, 1e-4)
    # sky-facing rim: upper and outer silhouette edges catch the cool sky glow
    rw = max(1, int(round(max(hpx * 0.006, 1.0))))
    up = np.zeros_like(a)
    up[rw:] = a[:-rw]
    edge_top = np.clip(a - up, 0, 1)
    lf = np.zeros_like(a)
    lf[:, rw:] = a[:, :-rw]
    rt = np.zeros_like(a)
    rt[:, :-rw] = a[:, rw:]
    edge_side = np.clip(a - np.minimum(lf, rt), 0, 1)
    rimk = np.clip(edge_top * 0.9 + edge_side * 0.45, 0, 1) * rim_amt
    rimk = rimk * (0.35 + 0.65 * sm)
    colL = colL * (1 - rimk[..., None] * 0.6) + rimc * rimk[..., None] * 0.6 * 1.0 + rimc * rimk[..., None] * 0.25
    # vertical light: the crown top catches more sky light than the lower tiers
    yy = np.clip((np.arange(h, dtype=np.float32)[:, None] - (bv - hpx - y0)) / max(hpx, 1), 0, 1)
    colL = colL * (1 + (0.1 - 0.3 * yy) * sm)[..., None]
    if warm is not None and warm_amt > 0:
        xx = (np.arange(w, dtype=np.float32)[None, :] - (bu - x0)) / max(hw0, 1)
        side = np.clip(0.5 + 0.6 * xx * warm_side, 0, 1) ** 1.6
        wk = (side * warm_amt)[..., None] * np.asarray(warm, np.float32)
        colL = colL + wk * (0.15 + 0.85 * sm[..., None])
        # the crisp cap rims on the lamp side catch the sodium light hard
        colL = colL + (rm * side * warm_amt)[..., None] * np.asarray(warm, np.float32) * 1.6
        # silhouette edge facing the lamp: thin warm rim
        rw2 = max(1, int(round(max(hpx * 0.004, 1.0))))
        sh_ = np.zeros_like(a)
        if warm_side > 0:
            sh_[:, :-rw2] = a[:, rw2:]
        else:
            sh_[:, rw2:] = a[:, :-rw2]
        el = np.clip(a - sh_, 0, 1) * (0.3 + 0.7 * sm)
        colL = colL + (el * warm_amt * 1.2)[..., None] * np.asarray(warm, np.float32)
    if fog is not None and fog_amt > 0:
        colL = colL * (1 - fog_amt) + np.asarray(fog, np.float32) * fog_amt
    cv.paint(x0, y0, np.clip(a, 0, 1), colL.astype(np.float32), z)
