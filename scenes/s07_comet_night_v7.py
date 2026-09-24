"""Round-9 painting helpers for s07_comet_night (Your Name comet night).

- tail_ribbons7: the split of the Your-Name tail. Instead of thin ruler-straight bands, the secondary
  structure is a set of broad, feathered, CURVED ribbons that peel off the main cyan fan and widen
  ~5x toward the frame edge: a magenta ribbon running right beside the cyan (bleeding into it, soft on
  both edges, with a sparkle-dust fringe), a cool blue outer ribbon, and a faint, very wide gold veil.
  Fine streamers at slightly different angles run through the whole tail.
- night_clouds7: flat, cut-out night cloud clusters (Your-Name yn_05): lobed silhouettes with a crisp,
  moonlit top edge, painted lighter top planes and undersides lost into the navy.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P
import s07_comet_night_v6 as V6


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _tail_coords(Wp, Hp, head, d, L, W, bend, reach):
    """Crop box + (u, across_bent) for the tail region. reach: max |across| in px to cover."""
    pts = []
    for u in np.linspace(0, 1.05, 14):
        for v in (-reach * 0.6, reach):
            acr = v * max(u, 0.05) + bend * L * u * u
            pts.append((head[0] + d[0] * u * L - d[1] * acr, head[1] + d[1] * u * L + d[0] * acr))
    pts = np.array(pts)
    x0, y0 = pts.min(0) - 0.03 * W
    x1, y1 = pts.max(0) + 0.03 * W
    box = (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, Wp)), int(min(y1, Hp)))
    bx0, by0, bx1, by1 = box
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    across = -px * d[1] + py * d[0]
    across = across - bend * L * np.clip(u, 0, None) ** 2
    return box, u, across


def tail_ribbons7(Wp, Hp, head, d, L, W, s, bend, w0, w1, wexp=0.75, seed=5, n_dust=2600):
    """Additive RGB (full plate) of the split secondary ribbons + streamers + sparkle dust fringe."""
    box, u, acr = _tail_coords(Wp, Hp, head, d, L, W, bend, 0.36 * W)
    bx0, by0, bx1, by1 = box
    uc = np.clip(u, 0, 1.2)
    wu = (w0 + (w1 - w0) * uc ** wexp) * W                  # main fan width
    start = _ss(0.0, 0.03, u)
    fall = start * np.clip(1.05 - u, 0, 1) ** 1.1 * (0.45 + 0.55 * np.exp(-uc / 0.35))
    img = np.zeros(u.shape + (3,), np.float32)

    def feather(a_, w_, seed_, fv=30.0):
        """striae across the ribbon (long along u)."""
        vn_ = a_ / np.maximum(w_, 1.0)
        return V6._striae(u * 0.9 + seed_ * 0.1, vn_, seed_, fu=1.8, fv=fv)

    def ribbon(c_frac, c_extra, hw0, hw1, seed_, soft_in=1.0, soft_out=1.0, fv=26.0, u_on=0.04):
        # centre: a fraction of the main fan width + an extra curve that peels the ribbon away
        cen = wu * c_frac + c_extra * L * uc ** 2 + 0.012 * W * P.fbm1d(uc.ravel(), 2.5, 2, seed_).reshape(uc.shape)
        hw = (hw0 + (hw1 - hw0) * uc ** 1.05) * W
        st = feather(acr - cen, hw, seed_, fv)
        dv = (acr - cen) / hw
        dv = dv + 0.35 * (st - 0.5)                           # ragged, feathered edges
        prof = np.where(dv < 0, np.exp(-(dv / soft_in) ** 2 * 1.6), np.exp(-(dv / soft_out) ** 2 * 1.6))
        body = prof * (0.45 + 0.9 * st) * _ss(u_on, u_on + 0.14, uc) * fall
        return body, cen, hw

    # magenta ribbon hugging the cyan fan's outer edge (inner edge bleeds into the cyan)
    mag, mcen, mhw = ribbon(0.3, 0.035, 0.003, 0.05, seed + 1, soft_in=1.35, soft_out=0.95)
    mag_col = np.array([1.0, 0.16, 0.72], np.float32)
    img += mag[..., None] * mag_col * 1.3
    # a pink-lavender inner blend (where the magenta meets the cyan it goes lilac, not muddy)
    blend = np.exp(-(((acr - (mcen - mhw * 0.7)) / (mhw * 0.6)) ** 2)) * _ss(0.06, 0.2, uc) * fall
    img += blend[..., None] * np.array([0.62, 0.42, 1.0], np.float32) * 0.2
    # cool blue ribbon beyond the magenta, peeling off with more curve
    blu, bcen, bhw = ribbon(0.3, 0.075, 0.004, 0.06, seed + 2, soft_in=1.0, soft_out=1.4, fv=22.0, u_on=0.1)
    blu = blu * _ss(0.0, 1.0, (acr - mcen) / (mhw + 1.0) + 0.3)
    img += blu[..., None] * np.array([0.2, 0.42, 1.0], np.float32) * 0.55
    # faint, very wide gold veil on the far outer side (low alpha, no line)
    gv_c = wu * 0.35 + 0.12 * L * uc ** 2
    gv_w = (0.006 + 0.1 * uc) * W
    gdv = (acr - gv_c) / gv_w
    gst = V6._striae(u * 0.5 + 3.0, gdv * 0.3, seed + 3, fu=1.2, fv=14.0)
    gold = np.exp(-gdv ** 2 * 1.2) * (0.5 + 0.8 * gst) * _ss(0.12, 0.35, uc) * fall
    img += gold[..., None] * np.array([1.0, 0.66, 0.3], np.float32) * 0.2

    # fine streamers at slightly different angles through the whole tail (hair-like jets)
    env_t = _ss(-0.9, -0.5, acr / np.maximum(wu, 1)) * np.exp(-np.clip(acr - mcen, 0, None) / (0.8 * mhw + 1)) * fall
    rng = np.random.default_rng(seed + 9)
    for k, ang in enumerate((-2.2, -0.8, 1.2, 2.6, 4.0)):
        ta = math.tan(math.radians(ang))
        a2 = acr - ta * uc * L
        sf = V6._striae(u * 0.4 + k, a2 / W, seed + 20 + k, fu=0.9, fv=900.0, blur_v=0.55, nv=4096)
        hair = np.clip((sf - 0.72) / 0.28, 0, 1) ** 1.6
        amt = rng.uniform(0.35, 0.6)
        col = np.array([0.55, 0.85, 1.0], np.float32) if k % 2 == 0 else np.array([0.7, 0.6, 1.0], np.float32)
        img += (hair * env_t * _ss(0.05, 0.25, uc))[..., None] * col * amt

    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img

    # sparkle-dust fringe along both magenta edges + in the blue ribbon
    rng = np.random.default_rng(seed + 31)
    uq = rng.uniform(0.05, 0.85, n_dust) ** 1.1
    wuq = (w0 + (w1 - w0) * uq ** wexp) * W
    cq = wuq * 0.3 + 0.035 * L * uq ** 2
    hq = (0.003 + 0.047 * uq ** 1.05) * W
    side = np.where(rng.random(n_dust) < 0.5, -1.0, 1.0)
    kind = rng.random(n_dust)
    off = np.where(kind < 0.75, side * hq * rng.uniform(0.6, 1.5, n_dust), rng.normal(0, 1.0, n_dust) * hq * 2.0 + hq)
    acq = cq + off + 0.0 * uq
    acq = acq + bend * L * uq ** 2
    x = head[0] + d[0] * uq * L - d[1] * acq
    y = head[1] + d[1] * uq * L + d[0] * acq
    cols = np.array([[1.0, 0.45, 0.95], [1.0, 0.85, 1.0], [0.6, 0.9, 1.0], [0.85, 0.6, 1.0], [1.0, 0.8, 0.55]],
                    np.float32)
    ci = rng.choice(5, n_dust, p=[0.34, 0.2, 0.22, 0.17, 0.07])
    m = (rng.random(n_dust) ** 5 * 1.6 + 0.05) * np.clip(1.0 - uq, 0.15, 1)
    acc = np.zeros((Hp, Wp, 3), np.float32)
    ok = (x >= 0) & (x < Wp - 1) & (y >= 0) & (y < Hp - 1)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m[ok] * cols[ci[ok], c])
    out += C.blur(acc, 0.5 * s + 0.2) * 3.2
    return out


# ----------------------------------------------------------------------------- night clouds
def night_clouds7(Wp, Hp, clusters, s, light_xy, seed=3, top_col=(0.74, 0.86, 1.02), body_col=(0.3, 0.42, 0.74),
                  sky_col=(0.05, 0.1, 0.25)):
    """Flat cut-out cloud clusters (yn_05). clusters: list of (cx, cy, width_px, height_px, amt).
    Returns straight RGBA (Hp, Wp, 4). Each cluster is a warped-noise mass inside a few flat, wide
    ellipses -> an irregular torn silhouette with breakaway fragments and holes; top edges of the mass and
    of its inner sub-masses are crisp and moonlit, bodies a flat painted mid value, the underside lost into
    the navy."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((Hp, Wp, 3), np.float32)
    A = np.zeros((Hp, Wp), np.float32)
    top_col = np.asarray(top_col, np.float32)
    body_col = np.asarray(body_col, np.float32)
    sky_col = np.asarray(sky_col, np.float32)
    for ci, (cx, cy, cw, ch, amt) in enumerate(clusters):
        pad = int(0.25 * cw)
        x0, x1 = int(max(cx - cw / 2 - pad, 0)), int(min(cx + cw / 2 + pad, Wp))
        y0, y1 = int(max(cy - ch * 1.2 - pad * 0.3, 0)), int(min(cy + ch * 1.0 + pad * 0.3, Hp))
        if x1 <= x0 or y1 <= y0:
            continue
        w_, h_ = x1 - x0, y1 - y0
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        # envelope: a few flat wide ellipses, slightly stepped
        env = np.zeros((h_, w_), np.float32)
        for k in range(int(rng.integers(3, 7))):
            q = rng.uniform(-0.42, 0.42)
            ex, ey = cx + q * cw, cy + rng.normal(0, 0.18) * ch
            rx = cw * rng.uniform(0.12, 0.3) * (1 - 0.4 * abs(q))
            ry = ch * rng.uniform(0.3, 0.6)
            env = np.maximum(env, np.exp(-(((xs - ex) / rx) ** 2 + ((ys - ey) / ry) ** 2)))
        sc = max(w_ / (0.13 * cw), 2)
        wx = (C.fbm(w_, h_, sc * 0.6, 3, seed=seed + 300 + ci) - 0.5) * 0.18 * cw
        wy = (C.fbm(w_, h_, sc * 0.6, 3, seed=seed + 400 + ci) - 0.5) * 0.25 * ch
        mx = np.clip(xs - x0 + wx, 0, w_ - 1).astype(np.float32)
        my = np.clip(ys - y0 + wy, 0, h_ - 1).astype(np.float32)
        n1 = C.fbm(w_, h_, sc, 6, gain=0.62, seed=seed + 100 + ci)
        n1 = cv2.remap(n1, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        envw = cv2.remap(env, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        n2 = C.fbm(w_, h_, sc * 3.0, 3, seed=seed + 200 + ci)
        n2 = 1 - np.abs(2 * n2 - 1)                          # billows: small puffy lobes
        field = envw + (1.1 * (n1 - 0.5) + 0.3 * (n2 - 0.6)) * (0.25 + 0.75 * _ss(0.0, 0.6, envw))
        field = cv2.GaussianBlur(field, (0, 0), 0.4 * s + 0.2)
        m = _ss(0.42, 0.47, field)                          # crisp cut-out edge
        # thin torn wisps / breakaway fragments around the mass
        wisp = _ss(0.3, 0.4, field) * _ss(0.52, 0.66, n2) * (1 - m) * 0.6
        a_ = np.clip(m + wisp, 0, 1)
        # moonlight from above: top boundaries of the mass AND of its inner sub-masses
        fs = cv2.GaussianBlur(field, (0, 0), 0.06 * ch + 1.0)
        dy = max(int(0.08 * ch), 2)
        gy = np.zeros_like(fs)
        gy[:-dy] = fs[dy:] - fs[:-dy]                        # >0 where the mass rises from above
        lit = _ss(0.02, 0.1, gy) * (0.75 + 0.5 * (n1 - 0.5))
        mb = (field > 0.45).astype(np.uint8)
        dist_top = np.zeros_like(field)
        run = np.full(w_, 1e4, np.float32)
        for r_ in range(h_):
            run = np.where(mb[r_] > 0, run + 1, 0)
            dist_top[r_] = run
        edge_lit = np.exp(-dist_top / (0.1 * ch + 1.5 * s)) * mb
        lit = np.clip(np.maximum(lit, edge_lit), 0, 1)
        lit = cv2.GaussianBlur(lit, (0, 0), sigmaX=1.5 * s + 0.5, sigmaY=0.4)
        lit = 0.5 * _ss(0.2, 0.45, lit) + 0.5 * lit            # painted: flat lit planes
        under = _ss(cy - ch * 0.2, cy + ch * 0.6, ys + 0.35 * ch * (n1 - 0.5))
        col = body_col + (top_col - body_col) * lit[..., None]
        col = col * (1 - 0.3 * under[..., None]) + sky_col * 0.3 * under[..., None]
        a_ = a_ * (1 - 0.8 * under * (1 - lit)) * amt
        rim = _ss(1.6 * s + 0.5, 0.4, dist_top) * (dist_top > 0) * (1 - under) * mb
        col = col + rim[..., None] * top_col * 0.15
        sub = A[y0:y1, x0:x1]
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1 - a_[..., None]) + col * a_[..., None]
        A[y0:y1, x0:x1] = sub + a_ * (1 - sub)
    rgb = rgb / np.maximum(A, 1e-5)[..., None] * (A > 1e-5)[..., None]
    return np.dstack([rgb, A]).astype(np.float32)


# ----------------------------------------------------------------------------- forest bands
def conifer_bands7(rgb, fmask, top_ss, depth, ss, s, Wp, th, light_x, seed, pal, haze_col, haze=0.35,
                   crown_col=(0.56, 0.72, 1.0)):
    """Painted forest bands (replaces the even, hatched conifer_bands): rows of tree-crown silhouettes
    stacked down the slope, but
      * rows get smaller / closer together and paler with distance (upper rows),
      * every row exists only in groves (broken along x) so the stacking never reads as scanlines,
      * irregular tops (clumped spires of varying height, rounded clumps, gaps),
      * lighter comet-lit crowns along the upper edge, a dark body that is LOST at the base (fades out
        instead of ending on a hard horizontal line),
      * snow clearings where the grove mask (fmask) opens up."""
    from s07_comet_night_hills import clumped_bumps
    rng = np.random.default_rng(seed)
    Hl = rgb.shape[0]
    xs2 = np.arange(Wp * ss, dtype=np.float32) / ss
    xr = np.arange(Wp, dtype=np.float32)
    t1 = top_ss.reshape(Wp, ss).mean(1)
    dmax = float(np.percentile(depth, 90))
    near = np.exp(-np.abs(xr - light_x) / (0.45 * Wp))[None, :]
    fm = np.clip(cv2.GaussianBlur(fmask, (0, 0), 0.8 * s + 0.3) * 1.2, 0, 1)
    fs, fl, fb = pal['forest_sh'], pal['forest_lit'], pal['forest_base']
    crown = np.asarray(crown_col, np.float32)
    haze_col = np.asarray(haze_col, np.float32)
    # row offsets below the ridge: small, tight rows far up the slope, bigger rows lower down
    offs, sizes = [], []
    o = 0.4 * th
    while o < dmax + th and len(offs) < 16:
        q = np.clip(o / max(dmax, 1.0), 0, 1)
        sz = th * (0.7 + 1.0 * q) * rng.uniform(0.85, 1.15)
        offs.append(o)
        sizes.append(sz)
        o += sz * rng.uniform(2.2, 3.4)
    nb = len(offs)
    for k in range(nb):
        fk = k / max(nb - 1, 1)
        sz = sizes[k]
        und = P.fbm1d(xs2 / Wp, 4.0, 3, seed + 13 * k) * 1.2 * sz + P.fbm1d(xs2 / Wp, 17.0, 2, seed + 5 * k) * 0.4 * sz
        base = np.repeat(t1, ss) + offs[k] + und
        tip, _ = clumped_bumps(xs2, Wp, sz * 1.25, seed + 29 * k, gap=-0.6)
        # a few rounded crown clumps mixed into the spires
        rc = np.clip(P.fbm1d(xs2 / Wp, Wp / (sz * 3.0), 2, seed + 71 * k), 0, 1) * sz * 0.9
        tip = np.maximum(tip, rc * (0.6 + 0.4 * _ss(0.0, 0.5, P.fbm1d(xs2 / Wp, 11.0, 2, seed + 83 * k))))
        line = base - tip
        # groves: each row is broken along x
        pres = _ss(-0.6, -0.3, P.fbm1d(xr / Wp, 5.0 + 3.0 * rng.random(), 3, seed + 97 * k))
        y0 = int(max(line.min() - 2, 0))
        y1 = int(min(Hl, base.max() + 4.5 * sz + 2))
        if y1 <= y0:
            continue
        ysl = np.arange(y0 * ss, y1 * ss, dtype=np.float32)[:, None] / ss
        m = np.clip((ysl - line[None, :]) * ss + 0.5, 0, 1)
        m = cv2.resize(m, (Wp, y1 - y0), interpolation=cv2.INTER_AREA)
        ln1 = line.reshape(Wp, ss).min(1)
        tp1 = tip.reshape(Wp, ss).max(1)
        b1 = base.reshape(Wp, ss).mean(1)
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        d = np.clip(yy - ln1[None, :], 0, None)
        vb = np.exp(-d / (1.3 * sz))
        body = fb * 0.7 + (fs - fb * 0.7) * vb[..., None] + (fl - fs) * (vb ** 2)[..., None] * 1.2 * near[..., None]
        # comet-lit crowns: pale on the upper side of the tallest crowns, patchy along the row
        tipness = np.clip(tp1 / (sz * 1.25 + 1e-3), 0, 1)[None, :]
        litn = 0.45 + 0.55 * _ss(-0.3, 0.4, P.fbm1d(xr / Wp, 13.0, 2, seed + 101 * k))[None, :]
        cap = np.exp(-d / (0.45 * sz + 0.5)) * (0.35 + 0.65 * tipness) * (0.5 + 0.5 * near) * litn
        body = body + (crown - body) * np.clip(cap * 0.9, 0, 0.8)[..., None]
        # aerial perspective: far (upper) rows paler, bluer, lower contrast
        hz = haze * (1 - fk) ** 1.1
        body = body * (1 - hz) + haze_col * hz
        # lost base: the row's body fades out below its base line (no hard horizontal edge)
        gap_k = (offs[k + 1] - offs[k]) if k + 1 < nb else 2.5 * sz
        lost = _ss(b1[None, :] + gap_k + 0.8 * sz, b1[None, :] + gap_k - 0.2 * sz, yy)
        a = m * fm[y0:y1] * pres[None, :] * lost
        rgb[y0:y1] = rgb[y0:y1] * (1 - a[..., None]) + body * a[..., None]
    return rgb
