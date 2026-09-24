"""s05_sakura helpers: the hand-designed foreground cherry branch (nearest layer, top-left corner).

Painted in 2D (screen space, supersampled): a tapered, kinked branch (thickness ~14 px -> ~2 px at 1080p),
side twigs forking at 30-45 degrees (some forking again), umbels of individual five-petal flowers with
notched petals, pink bases, dark centres and stamens on the twig tips and on spurs along the limb (they hide
the wood in places), a few buds.  Backlit by the sun: dark wood with a warm rim, translucent glowing petal
edges.  Slightly defocused (nearest plane).
"""
import math
import numpy as np
import cv2
from numba import njit

from s05_sakura_raster import sstep, hash2


# ----------------------------------------------------------------------------- structure

class Branch:
    def __init__(self):
        self.segs = []      # x0 y0 r0 x1 y1 r1 arc0 len
        self.flowers = []   # x y R rot squash tilt tone kind(0 flower, 1 bud, 2 back-facing) depth
        self.stems = []     # x0 y0 x1 y1 width


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def build(W, H, seed=7):
    """Branch geometry in frame pixels (base camera position)."""
    rng = np.random.default_rng(seed)
    B = Branch()
    u = H / 1080.0          # 1 px at 1080p
    arc = [0.0]

    def limb(p, d, n, seg_len, r0, r1, level, bend_sign):
        """kinked limb: n straight-ish segments that change angle at nodes."""
        pts = [np.array(p, np.float64)]
        dirs = []
        d = d / np.linalg.norm(d)
        for k in range(n):
            if k > 0:
                # kink: alternate the bend (zig-zag growth between buds) + gravity droop
                ang = math.radians(rng.uniform(9, 22)) * bend_sign * (1 if k % 2 else -1)
                d = _rot(d, ang)
                d = d + np.array([0.0, 0.035 if level == 0 else 0.08])
                d = d / np.linalg.norm(d)
            L = seg_len * rng.uniform(0.8, 1.15) * (0.9 ** k)
            # slight curvature inside a segment
            mid = pts[-1] + d * L * 0.5 + _rot(d, math.pi / 2) * rng.normal(0, 0.02 * L)
            end = pts[-1] + d * L
            pts.append(mid)
            pts.append(end)
            dirs.append(d.copy())
        rr = np.linspace(r0, r1, len(pts))
        for k in range(len(pts) - 1):
            L = float(np.linalg.norm(pts[k + 1] - pts[k]))
            B.segs.append((pts[k][0], pts[k][1], rr[k], pts[k + 1][0], pts[k + 1][1], rr[k + 1], arc[0], L))
            arc[0] += L
        return pts, rr, dirs

    def cluster(c, n, Rf, out_dir, depth0):
        """umbel of n flowers (+ buds) around c; pedicels radiate from a common point."""
        c = np.asarray(c, np.float64)
        for i in range(n):
            a = rng.uniform(-1.2, 1.2) + math.atan2(out_dir[1], out_dir[0])
            dist = Rf * rng.uniform(0.3, 1.35) * (1.0 + 0.08 * n)
            q = c + np.array([math.cos(a), math.sin(a)]) * dist
            R = Rf * rng.uniform(0.85, 1.15)
            kind = 0
            if rng.random() < 0.18:
                kind = 2       # seen from the back / side: lilac, less structure
            B.stems.append((c[0], c[1], q[0], q[1], max(0.7 * u, 1.0 * u)))
            B.flowers.append((q[0], q[1], R, rng.uniform(0, 6.283), rng.uniform(0.55, 1.0), rng.uniform(0, 3.14),
                              rng.uniform(-1, 1), kind, depth0 + rng.uniform(0, 1)))
        for i in range(int(rng.integers(0, 3))):
            a = rng.uniform(0, 6.283)
            q = c + np.array([math.cos(a), math.sin(a)]) * Rf * rng.uniform(0.8, 1.6)
            B.stems.append((c[0], c[1], q[0], q[1], 0.8 * u))
            B.flowers.append((q[0], q[1], Rf * 0.36, a, 0.6, a, 0.0, 1, depth0 + rng.uniform(0, 1)))

    Rf = 0.0135 * H
    # main limb enters from beyond the top-left edge and hangs across toward the sun side of the sky
    p0 = np.array([-0.3 * W, -0.075 * H])
    d0 = np.array([1.0, 0.13])
    pts, rr, dirs = limb(p0, d0, 6, 0.155 * W, 9.5 * u, 1.8 * u, 0, 1)
    nodes = [pts[2 * k] for k in range(1, len(pts) // 2 + 1)]
    # side twigs at the nodes, forking 30-45 degrees off the limb, alternating sides
    side = 1
    for k, nd in enumerate(nodes[:-1]):
        dpar = dirs[min(k + 1, len(dirs) - 1)]
        rpar = rr[2 * (k + 1)]
        ntw = 2 if k >= 1 else 1
        for j in range(ntw):
            side = -side
            ang = math.radians(rng.uniform(30, 45)) * side
            dt = _rot(dpar, ang)
            if dt[1] < -0.55:             # keep twigs from shooting straight up
                dt = _rot(dpar, -ang)
            base = nd if j == 0 else nd + dpar * rng.uniform(0.2, 0.45) * 0.1 * W
            tl = rng.uniform(0.045, 0.085) * W * (1.0 - 0.1 * k)
            tp, trr, tdirs = limb(base, dt, 2, tl * 0.55, rpar * 0.55, 1.2 * u, 1, -side)
            tip = tp[-1]
            cluster(tip + tdirs[-1] * Rf * 0.6, int(rng.integers(7, 12)), Rf, tdirs[-1], 1.0)
            if rng.random() < 0.6:
                cluster(tp[2], int(rng.integers(3, 6)), Rf * 0.9, _rot(tdirs[0], 1.5 * side), 0.8)
            # secondary fork near the twig's middle
            if rng.random() < 0.75:
                ang2 = math.radians(rng.uniform(30, 45)) * (-side)
                d2 = _rot(tdirs[0], ang2)
                b2 = tp[2]
                sp, srr, sdirs = limb(b2, d2, 1, tl * rng.uniform(0.35, 0.55), trr[2] * 0.7, 1.0 * u, 2, side)
                cluster(sp[-1] + sdirs[-1] * Rf * 0.5, int(rng.integers(6, 10)), Rf * 0.95, sdirs[-1], 0.5)
    # spur clusters directly on the limb (hide the wood in places)
    for k in range(1, len(pts) - 1):
        if k < 5 or rng.random() > 0.7:
            continue
        nd = pts[k]
        off = _rot(dirs[min(k // 2, len(dirs) - 1)], math.pi / 2 * (1 if rng.random() < 0.6 else -1))
        cluster(nd + off * Rf * 0.5, int(rng.integers(5, 10)), Rf * 1.05, off, 2.0)
    # terminal cluster
    cluster(pts[-1] + dirs[-1] * Rf * 0.7, 11, Rf * 1.05, dirs[-1], 2.5)
    return B


# ----------------------------------------------------------------------------- raster

@njit(cache=True, fastmath=True)
def _draw_wood(buf, segs, ox, oy, ss, lx, ly):
    H, W = buf.shape[0], buf.shape[1]
    for i in range(segs.shape[0]):
        ax = (segs[i, 0] - ox) * ss
        ay = (segs[i, 1] - oy) * ss
        ar = segs[i, 2] * ss
        bx = (segs[i, 3] - ox) * ss
        by = (segs[i, 4] - oy) * ss
        br = segs[i, 5] * ss
        a0 = segs[i, 6]
        Ls = segs[i, 7]
        rm = max(ar, br) + 2
        x0 = max(0, int(min(ax, bx) - rm))
        x1 = min(W, int(max(ax, bx) + rm + 1))
        y0 = max(0, int(min(ay, by) - rm))
        y1 = min(H, int(max(ay, by) + rm + 1))
        ex = bx - ax
        ey = by - ay
        L2 = ex * ex + ey * ey + 1e-6
        ln = math.sqrt(L2)
        nx = -ey / ln
        ny = ex / ln
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5 - ax
                py = y + 0.5 - ay
                t = (px * ex + py * ey) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = px - t * ex
                qy = py - t * ey
                rr = ar + (br - ar) * t
                d = math.sqrt(qx * qx + qy * qy)
                cov = min(max(rr + 0.5 - d, 0.0), 1.0)
                if cov <= 0.0:
                    continue
                sg = (qx * nx + qy * ny) / max(rr, 0.5)       # -1..1 across the limb
                sg = min(max(sg, -1.0), 1.0)
                facing = -(nx * lx + ny * ly) * sg              # >0 on the side toward the light
                # dark backlit wood, cool bounce on the far side, hard warm rim toward the sun
                v = 0.1 + 0.06 * (1 - abs(sg))
                r = v * 0.9
                g = v * 0.68
                b = v * 0.85
                rim = sstep(0.55, 0.85, abs(sg)) * sstep(0.0, 0.3, facing)
                r += rim * 0.95
                g += rim * 0.62
                b += rim * 0.42
                cool = sstep(0.6, 0.95, abs(sg)) * sstep(0.0, 0.3, -facing) * 0.18
                r += cool * 0.5
                g += cool * 0.5
                b += cool * 0.85
                # horizontal lenticel bands across the limb
                arc = (a0 + t * Ls * 1.0) / ss * ss
                bnd = hash2(int(math.floor((a0 + t * Ls) / (3.2))), i, 5)
                if bnd > 0.7:
                    r *= 0.75
                    g *= 0.75
                    b *= 0.8
                a = cov
                buf[y, x, 0] = buf[y, x, 0] * (1 - a) + r * a
                buf[y, x, 1] = buf[y, x, 1] * (1 - a) + g * a
                buf[y, x, 2] = buf[y, x, 2] * (1 - a) + b * a
                buf[y, x, 3] = buf[y, x, 3] * (1 - a) + a


@njit(cache=True, fastmath=True)
def _draw_stems(buf, st, ox, oy, ss):
    H, W = buf.shape[0], buf.shape[1]
    for i in range(st.shape[0]):
        ax = (st[i, 0] - ox) * ss
        ay = (st[i, 1] - oy) * ss
        bx = (st[i, 2] - ox) * ss
        by = (st[i, 3] - oy) * ss
        w = st[i, 4] * ss * 0.5
        x0 = max(0, int(min(ax, bx) - w - 2))
        x1 = min(W, int(max(ax, bx) + w + 3))
        y0 = max(0, int(min(ay, by) - w - 2))
        y1 = min(H, int(max(ay, by) + w + 3))
        ex = bx - ax
        ey = by - ay
        L2 = ex * ex + ey * ey + 1e-6
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5 - ax
                py = y + 0.5 - ay
                t = min(max((px * ex + py * ey) / L2, 0.0), 1.0)
                qx = px - t * ex
                qy = py - t * ey
                d = math.sqrt(qx * qx + qy * qy)
                cov = min(max(w + 0.5 - d, 0.0), 1.0)
                if cov <= 0.0:
                    continue
                r, g, b = 0.42 + 0.3 * t, 0.2 + 0.12 * t, 0.26 + 0.1 * t
                buf[y, x, 0] = buf[y, x, 0] * (1 - cov) + r * cov
                buf[y, x, 1] = buf[y, x, 1] * (1 - cov) + g * cov
                buf[y, x, 2] = buf[y, x, 2] * (1 - cov) + b * cov
                buf[y, x, 3] = buf[y, x, 3] * (1 - cov) + cov


@njit(cache=True, fastmath=True)
def _draw_flowers(buf, fl, ox, oy, ss, sunx, suny):
    H, W = buf.shape[0], buf.shape[1]
    for i in range(fl.shape[0]):
        cx = (fl[i, 0] - ox) * ss
        cy = (fl[i, 1] - oy) * ss
        R = fl[i, 2] * ss
        rot = fl[i, 3]
        sq = fl[i, 4]
        tl = fl[i, 5]
        tone = fl[i, 6]
        kind = int(fl[i, 7])
        x0 = max(0, int(cx - R - 2))
        x1 = min(W, int(cx + R + 3))
        y0 = max(0, int(cy - R - 2))
        y1 = min(H, int(cy + R + 3))
        ct = math.cos(tl)
        st_ = math.sin(tl)
        # toward-the-sun direction in screen (for the translucent glow on the facing petal edges)
        sdx = (sunx - ox) * ss - cx
        sdy = (suny - oy) * ss - cy
        sdl = math.sqrt(sdx * sdx + sdy * sdy) + 1e-6
        sdx /= sdl
        sdy /= sdl
        lum = 0.92 + 0.1 * tone
        for y in range(y0, y1):
            for x in range(x0, x1):
                dx = x + 0.5 - cx
                dy = y + 0.5 - cy
                # foreshortening (tilted flower)
                u = dx * ct + dy * st_
                v = (-dx * st_ + dy * ct) / sq
                rho = math.sqrt(u * u + v * v)
                if kind == 1:
                    # bud: small teardrop, deep pink
                    q = rho / R
                    cov = min(max((1.0 - q) * R + 0.5, 0.0), 1.0)
                    if cov <= 0.0:
                        continue
                    r, g, b = 0.93, 0.45 + 0.2 * (1 - q), 0.6 + 0.1 * (1 - q)
                else:
                    th = math.atan2(v, u) - rot
                    sec = 2.0 * math.pi / 5.0
                    k = math.floor(th / sec + 0.5)
                    ph = th - k * sec
                    cph = math.cos(ph * 2.25)
                    if cph <= 0.0:
                        rp = 0.3 * R
                    else:
                        rp = R * (0.3 + 0.7 * cph ** 0.45)
                    rp *= 1.0 - 0.16 * math.exp(-(ph / 0.075) ** 2)          # notch at the petal tip
                    edge = (rp - rho) * sq ** 0.5
                    cov = min(max(edge + 0.5, 0.0), 1.0)
                    if cov <= 0.0:
                        continue
                    t = rho / R
                    # petal colour: pale translucent white-pink tips, pink base, dark centre + stamens
                    base = 1.0 - sstep(0.12, 0.55, t)
                    r = 1.02 * lum
                    g = (0.9 - 0.24 * base) * lum
                    b = (0.93 - 0.12 * base) * lum
                    # faint vein down the petal middle
                    vein = math.exp(-(ph / 0.06) ** 2) * sstep(0.2, 0.5, t) * (1 - sstep(0.7, 0.95, t)) * 0.06
                    g -= vein
                    b -= vein * 0.5
                    # overlap shading: the left side of each petal tucks under its neighbour
                    ov = sstep(0.1, 0.55, -ph) * 0.07
                    r -= ov * 0.6
                    g -= ov
                    b -= ov * 0.5
                    # painted outline at the petal edge (lilac-pink)
                    ol = 1.0 - sstep(0.4, 1.6, edge)
                    r += (0.86 - r) * ol * 0.45
                    g += (0.58 - g) * ol * 0.45
                    b += (0.72 - b) * ol * 0.45
                    # translucent backlit glow on the sun-facing rim
                    fac = (u * sdx + v * sdy) / (rho + 1e-3)
                    gl = sstep(0.55, 0.95, t) * sstep(0.0, 0.8, fac) * 0.22
                    r += gl
                    g += gl * 0.85
                    b += gl * 0.65
                    if kind == 2:
                        # back view: lilac, no centre
                        r, g, b = r * 0.86, g * 0.78, b * 0.93
                    else:
                        c = 1.0 - sstep(0.1, 0.17, t)
                        r += (0.78 - r) * c
                        g += (0.24 - g) * c
                        b += (0.4 - b) * c
                        # stamens: little dots on a ring
                        sa = math.atan2(v, u) * 12.0 / (2 * math.pi)
                        sf = sa - math.floor(sa)
                        stm = sstep(0.22, 0.26, t) * (1 - sstep(0.3, 0.34, t)) * (1 - sstep(0.25, 0.4, abs(sf - 0.5)))
                        r += (1.0 - r) * stm * 0.8
                        g += (0.78 - g) * stm * 0.8
                        b += (0.5 - b) * stm * 0.8
                        fil = (1 - sstep(0.15, 0.3, abs(sf - 0.5))) * sstep(0.14, 0.18, t) * (1 - sstep(0.22, 0.25, t))
                        r += (0.85 - r) * fil * 0.5
                        g += (0.4 - g) * fil * 0.5
                        b += (0.5 - b) * fil * 0.5
                buf[y, x, 0] = buf[y, x, 0] * (1 - cov) + r * cov
                buf[y, x, 1] = buf[y, x, 1] * (1 - cov) + g * cov
                buf[y, x, 2] = buf[y, x, 2] * (1 - cov) + b * cov
                buf[y, x, 3] = buf[y, x, 3] * (1 - cov) + cov


def render(W, H, sun_xy, seed=7, defocus=0.0006, top=0.1):
    """-> (rgba plate (h, w, 4) straight alpha, ox, oy, pivot) in frame pixels.  `top`: headroom above the
    frame (fraction of H) - round 7 tilt-up reveals the branch above the frame edge."""
    B = build(W, H, seed)
    ox, oy = -0.36 * W, -top * H
    pw, ph = int(1.2 * W), int((0.52 + top) * H)
    ss = 2
    buf = np.zeros((ph * ss, pw * ss, 4), np.float32)
    segs = np.array(B.segs, np.float64)
    fl = np.array(B.flowers, np.float64)
    st = np.array(B.stems, np.float64)
    # light direction (screen) : from the sun toward the branch -> rim on the sun-facing side
    lx, ly = -0.75, -0.66
    _draw_wood(buf, segs, ox, oy, ss, lx, ly)
    order = np.argsort(fl[:, 8])
    _draw_stems(buf, st, ox, oy, ss)
    _draw_flowers(buf, fl[order], ox, oy, ss, sun_xy[0], sun_xy[1])
    # downsample (the buffer is premultiplied: every draw is a premultiplied 'over')
    pm = cv2.resize(buf, (pw, ph), interpolation=cv2.INTER_AREA)
    sig = defocus * W
    if sig > 0.3:
        pm = cv2.GaussianBlur(pm, (0, 0), sig)
    a = pm[..., 3:4]
    rgb = pm[..., :3] / np.maximum(a, 1e-4)
    rgba = np.concatenate([rgb, a], -1).astype(np.float32)
    return rgba, int(round(ox)), int(round(oy))


def decorate_canopy(rgba, z, f, W, row0, horizon, seed=0):
    """Painterly pass on a near tree card (straight-alpha RGBA, in place semantics -> returns new array):
    * sky holes punched through the interior of the upper canopy (light shows through),
    * individual five-petal flowers stamped along the LIT silhouette edges (up-left, toward the sun),
      pushed slightly outward so they break the outline."""
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(seed)
    h, w = rgba.shape[:2]
    a = rgba[..., 3].copy()
    col = rgba[..., :3]
    pink = ((col[..., 0] - col[..., 1]) > 0.05) & (col[..., 2] > col[..., 1] + 0.02)   # blossom, not lit bark
    R = float(np.clip(f * 0.02 / z, 0.0022 * W, 0.0055 * W))
    # ---- sky holes: small ragged gaps a little inside the silhouette (the sky shows through between the
    #      flower bunches); each gap = a cluster of small overlapping round voids, never a clean circle
    dist = distance_transform_edt(a > 0.5)
    yy = np.arange(h)[:, None] + row0
    wood = (a > 0.5) & ~pink
    dwood = distance_transform_edt(~wood) if wood.any() else np.full(a.shape, 1e9)
    band_ = (dist > 2.5 * R) & (dist < 8 * R) & pink & (yy < horizon - 0.06 * W) & (dwood > 5 * R)
    iy, ix = np.nonzero(band_)
    if len(iy):
        n = int(min(len(iy) / (26 * R) ** 2, 60))
        pick = rng.choice(len(iy), size=min(n, len(iy)), replace=False)
        m = np.zeros((h * 2, w * 2), np.uint8)
        for k in pick:
            cx, cy = ix[k], iy[k]
            nv = int(rng.integers(4, 10))
            for j in range(nv):
                rr = R * rng.uniform(0.5, 1.25)
                ox_, oy_ = rng.normal(0, R * 1.3, 2)
                cv2.circle(m, (int((cx + ox_) * 2), int((cy + oy_) * 2)), max(1, int(rr * 2)), 255, -1, cv2.LINE_AA)
        hole = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        hole *= np.clip((dwood - 2.0 * R) / R, 0, 1)
        a = a * (1 - hole)
    # ---- lit-edge flowers
    d = max(2, int(round(0.0016 * W)))
    from s05_sakura_canopy import shift_rep
    sh = shift_rep(a, d, d)             # alpha a few px toward the sun (up-left); edges replicated
    lum = col.mean(-1)
    edge = (a > 0.85) & (sh < 0.25) & pink & (lum > 0.62)
    ey, ex = np.nonzero(edge)
    fl = []
    if len(ey):
        cell = 1.7 * R
        key = (ey // cell).astype(np.int64) * 100000 + (ex // cell).astype(np.int64)
        perm = rng.permutation(len(ey))
        _, first = np.unique(key[perm], return_index=True)
        sel = perm[first]
        # keep flowers in small groups along the edge (not a bead line): modulate by a coarse random field
        grp = rng.random(1024)
        gk = ((ey[sel] // (cell * 5)).astype(np.int64) * 31 + (ex[sel] // (cell * 5)).astype(np.int64)) % 1024
        sel = sel[(grp[gk] < 0.45) & (rng.random(len(sel)) < 0.85)]
        for k in sel:
            x = ex[k] - R * rng.uniform(0.1, 0.5)
            y = ey[k] - R * rng.uniform(0.1, 0.5)
            fl.append((x, y, R * rng.uniform(0.75, 1.2), rng.uniform(0, 6.283), rng.uniform(0.55, 1.0),
                       rng.uniform(0, 3.14), rng.uniform(-0.2, 1.0), 2 if rng.random() < 0.12 else 0,
                       rng.random()))
        # a sprinkling one row inside the edge (clusters, not a single bead line)
        for k in sel[rng.random(len(sel)) < 0.5]:
            x = ex[k] + R * rng.uniform(0.6, 1.4)
            y = ey[k] + R * rng.uniform(0.6, 1.4)
            fl.append((x, y, R * rng.uniform(0.7, 1.0), rng.uniform(0, 6.283), rng.uniform(0.55, 1.0),
                       rng.uniform(0, 3.14), rng.uniform(-0.4, 0.6), 0, rng.random() - 1))
    # ---- painterly flower-cluster dabs over the dome shading (varied bunch sizes, crisp sun crescents)
    from s05_sakura_paint import paint_clusters
    col = paint_clusters(col, a, pink, R, (-0.66, -0.75), rng)
    # ---- flower texture inside the masses: small bunches of individual blossoms, dense and near-white on
    #      the lit tops, sparse and rose-tinted in the shade (reads as flowers, not as smooth domes)
    lum = col.mean(-1)
    cellf = 1.5 * R
    gh, gw = int(h / cellf) + 1, int(w / cellf) + 1
    jy = (np.arange(gh)[:, None] + rng.random((gh, gw))) * cellf
    jx = (np.arange(gw)[None, :] + rng.random((gh, gw))) * cellf
    jy = np.clip(jy, 0, h - 1).astype(int).ravel()
    jx = np.clip(jx, 0, w - 1).astype(int).ravel()
    okf = (a[jy, jx] > 0.97) & pink[jy, jx]
    jy, jx = jy[okf], jx[okf]
    if len(jy):
        lv = lum[jy, jx]
        litk = np.clip((lv - 0.55) / 0.3, 0, 1)
        cl_ = cv2.resize(rng.random((gh // 4 + 2, gw // 4 + 2)).astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)[jy, jx]
        pr_ = (0.02 + 0.35 * litk) * np.clip((cl_ - 0.45) * 3.0, 0, 1)
        keep = rng.random(len(jy)) < pr_
        jy, jx, litk = jy[keep], jx[keep], litk[keep]
        c0 = col[jy, jx]
        lift = (0.05 + 0.13 * litk)[:, None] * rng.uniform(0.6, 1.3, (len(jy), 1))
        cc = c0 * (1 + lift) + np.array([0.04, 0.02, 0.02]) * litk[:, None]
        cc = np.minimum(cc, np.array([1.1, 1.02, 1.0]))
        n_ = len(jy)
        flt = np.stack([jx + rng.normal(0, 0.3 * R, n_), jy + rng.normal(0, 0.3 * R, n_),
                        R * rng.uniform(0.55, 0.95, n_), rng.uniform(0, 6.283, n_), rng.uniform(0.55, 1.0, n_),
                        rng.uniform(0, 3.14, n_), cc[:, 0], cc[:, 1], cc[:, 2], 0.25 * litk], 1)
        flt = flt[np.argsort(flt[:, 1])]
        from s05_sakura_canopy import draw_florets
        big = cv2.resize(np.dstack([col, a]).astype(np.float32), (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        flt2 = flt.copy()
        flt2[:, 0:3] *= 2
        draw_florets(big, flt2.astype(np.float64))
        sm = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
        col = sm[..., :3].copy()
    # ---- rim light on silhouettes facing the sun (backlit petals glow where the canopy meets the sky)
    dr = max(2, int(round(0.0035 * W)))
    shs = shift_rep(a, dr, dr)
    rim = np.clip(a - shs, 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), max(0.6, 0.0008 * W)) * a
    gy, gx = np.mgrid[0:h, 0:w]
    cells = rng.random((h // 40 + 2, w // 40 + 2)).astype(np.float32)
    grp_ = cv2.resize(cells, (w, h), interpolation=cv2.INTER_CUBIC)
    rim *= np.clip((grp_ - 0.35) * 3.0, 0, 1) * (gy + row0 < horizon + 0.02 * W)
    rim = np.clip(rim * 1.4, 0, 1)[..., None]
    col = col + (np.array([1.08, 0.94, 0.95], np.float32) - col) * rim * 0.85
    pm = np.concatenate([col * a[..., None], a[..., None]], -1).astype(np.float32)
    if fl:
        fl = np.array(fl, np.float64)
        fl = fl[np.argsort(fl[:, 8])]
        big = cv2.resize(pm, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        _draw_flowers(big, fl, 0.0, 0.0, 2.0, -1e4, -1e4)
        pm = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    a = pm[..., 3:4]
    return np.concatenate([pm[..., :3] / np.maximum(a, 1e-4), a], -1).astype(np.float32)
