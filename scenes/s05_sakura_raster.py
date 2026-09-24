"""s05_sakura helpers: numba rasterizers.

* hash / value noise usable inside numba kernels (world-space textures: dapples, gravel, petals)
* dome/capsule z-buffer for painted blossom canopies (spheres = blossom clumps / florets, capsules = branches)
* painterly shading of the z-buffer (lit tops, lavender undersides, crevice AO, translucent rims, aerial haze)
* falling-petal renderer (tumbling ellipses with depth-of-field)
"""
import math
import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------------- noise

@njit(cache=True, inline='always')
def hash2(ix, iy, seed):
    n = np.int64(ix) * np.int64(374761393) + np.int64(iy) * np.int64(668265263) + np.int64(seed) * np.int64(1442695041)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n ^ (n >> np.int64(16))
    return (n & np.int64(0xFFFFFF)) / 16777216.0


@njit(cache=True, inline='always')
def vnoise(x, y, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    a = hash2(ix, iy, seed)
    b = hash2(ix + 1, iy, seed)
    c = hash2(ix, iy + 1, seed)
    d = hash2(ix + 1, iy + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


@njit(cache=True, inline='always')
def fbm2(x, y, seed, octaves):
    s = 0.0
    amp = 0.5
    tot = 0.0
    for o in range(octaves):
        s += amp * vnoise(x, y, seed + o * 31)
        tot += amp
        x = x * 2.03 + 17.1
        y = y * 2.03 + 5.3
        amp *= 0.5
    return s / tot


@njit(cache=True, inline='always')
def sstep(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3 - 2 * t)


@njit(cache=True, inline='always')
def worley(x, y, seed):
    """F1, F2 distances to jittered cell points (cell = 1)."""
    ix = math.floor(x)
    iy = math.floor(y)
    d1 = 9.0
    d2 = 9.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            cx = ix + dx
            cy = iy + dy
            px = cx + 0.15 + 0.7 * hash2(cx, cy, seed)
            py = cy + 0.15 + 0.7 * hash2(cx, cy, seed + 1)
            d = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if d < d1:
                d2 = d1
                d1 = d
            elif d < d2:
                d2 = d
    return d1, d2


@njit(cache=True, inline='always')
def flower_tex(px, py, wsz, i):
    """Painted blossom texture at screen px (ss px), wsz = metres per pixel.
    Returns (cluster, sparkle): cluster -1..1 soft value variation of flower bunches (~0.14 m),
    sparkle 0..1 coverage of individual five-petal flowers (~0.04 m) that catch the light."""
    cl = 0.0
    fr_ = (i * 0.618) % 1.0
    cpx = (0.08 + 0.15 * fr_) / wsz
    k = sstep(2.0, 6.0, cpx)
    if k > 0.0:
        d1, d2 = worley(px / cpx + i * 0.618, py / cpx * 1.2 + i * 0.37, 71)
        # hard-edged cluster cells (flat-ish lit tops, dark seams) instead of round soft dabs
        cl = k * ((1.0 - sstep(0.3, 0.62, d1)) * 1.1 - 0.4 - 0.18 * (1.0 - sstep(0.02, 0.06, d2 - d1)))
    sp = 0.0
    fpx = 0.042 / wsz
    k2 = sstep(2.2, 4.5, fpx)
    if k2 > 0.0:
        gx = px / fpx + i * 0.31
        gy = py / fpx + i * 0.77
        ix = math.floor(gx)
        iy = math.floor(gy)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx = ix + dx
                cy = iy + dy
                h = hash2(cx, cy, 81)
                if h > 0.55:
                    continue
                fx = cx + 0.2 + 0.6 * hash2(cx, cy, 82)
                fy = cy + 0.2 + 0.6 * hash2(cx, cy, 83)
                rx = gx - fx
                ry = gy - fy
                d = math.sqrt(rx * rx + ry * ry)
                th = math.atan2(ry, rx) + h * 20.0
                rr = 0.42 * (0.72 + 0.28 * abs(math.cos(2.5 * th)))
                c = 1.0 - sstep(rr - 0.9 / fpx - 0.03, rr + 0.9 / fpx, d)
                if c > sp:
                    sp = c
        sp *= k2
    return cl, sp


# ----------------------------------------------------------------------------- primitive z-buffer
# primitive table columns
K_KIND, K_X0, K_Y0, K_Z0, K_R0, K_X1, K_Y1, K_Z1, K_R1, K_AY, K_RZ0, K_RZ1, K_PAR, K_TREE, K_MAT, K_TONE = range(16)
NP = 16
# materials: 0 blossom clump, 1 floret, 2 bark


@njit(cache=True, parallel=True, fastmath=True)
def zbuffer(Hs, Ws, P, band_lists, band_ptr, band_h, zb, ib, hy_ss, f_ss, offy, yg, bark_bias):
    nb = band_ptr.shape[0] - 1
    for b in prange(nb):
        yb0 = b * band_h
        yb1 = min(Hs, yb0 + band_h)
        for q in range(band_ptr[b], band_ptr[b + 1]):
            i = band_lists[q]
            if P[i, K_KIND] == 0.0:
                cx = P[i, K_X0]
                cy = P[i, K_Y0]
                z0 = P[i, K_Z0]
                r = P[i, K_R0]
                ry = r * P[i, K_AY]
                rz = P[i, K_RZ0]
                x0 = max(0, int(cx - r - 1))
                x1 = min(Ws, int(cx + r + 2))
                y0 = max(yb0, int(cy - ry - 1))
                y1 = min(yb1, int(cy + ry + 2))
                ir = 1.0 / r
                iry = 1.0 / ry
                flower = P[i, K_MAT] == 1.0 and r > 3.0
                scal = P[i, K_MAT] == 0.0 and r > 6.0
                ph = P[i, K_TONE] * 2.7
                nlob = 3.0 + math.floor(hash2(i, 1, 55) * 3.0) + 0.5
                for y in range(y0, y1):
                    dy = (y + 0.5 - cy) * iry
                    dy2 = dy * dy
                    if dy2 >= 1.0:
                        continue
                    for x in range(x0, x1):
                        dx = (x + 0.5 - cx) * ir
                        d2 = dx * dx + dy2
                        if d2 >= 1.0:
                            continue
                        if flower:
                            th = math.atan2(dy, dx)
                            if r > 12.0:
                                # big floret = a bunch of flowers: irregular scalloped outline (more lobes)
                                lob = 0.74 + 0.26 * math.sqrt(abs(math.cos(3.5 * th + ph))) *                                     (0.7 + 0.3 * math.sin(1.0 * th + ph * 1.7))
                            else:
                                lob = 0.7 + 0.3 * math.sqrt(abs(math.cos(2.5 * th + ph)))
                            d2 = d2 / (lob * lob)
                            if d2 >= 1.0:
                                continue
                        elif scal:
                            th = math.atan2(dy, dx)
                            lob = 0.84 + 0.16 * math.sqrt(abs(math.cos(nlob * th + ph)))
                            d2 = d2 / (lob * lob)
                            if d2 >= 1.0:
                                continue
                        z = z0 - rz * math.sqrt(1.0 - d2)
                        if (hy_ss - (y + 0.5 + offy)) / f_ss * z < yg:
                            continue
                        if z < zb[y, x]:
                            zb[y, x] = z
                            ib[y, x] = i
            else:
                ax, ay_, az, ar = P[i, K_X0], P[i, K_Y0], P[i, K_Z0], P[i, K_R0]
                bx, by, bz, br = P[i, K_X1], P[i, K_Y1], P[i, K_Z1], P[i, K_R1]
                rz0, rz1 = P[i, K_RZ0], P[i, K_RZ1]
                rm = max(ar, br)
                x0 = max(0, int(min(ax, bx) - rm - 1))
                x1 = min(Ws, int(max(ax, bx) + rm + 2))
                y0 = max(yb0, int(min(ay_, by) - rm - 1))
                y1 = min(yb1, int(max(ay_, by) + rm + 2))
                ex = bx - ax
                ey = by - ay_
                L2 = ex * ex + ey * ey + 1e-6
                for y in range(y0, y1):
                    py = y + 0.5 - ay_
                    for x in range(x0, x1):
                        px = x + 0.5 - ax
                        t = (px * ex + py * ey) / L2
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        qx = px - t * ex
                        qy = py - t * ey
                        rr = ar + (br - ar) * t
                        d2 = (qx * qx + qy * qy) / (rr * rr)
                        if d2 >= 1.0:
                            continue
                        z = az + (bz - az) * t - (rz0 + (rz1 - rz0) * t) * math.sqrt(1.0 - d2) + bark_bias
                        if (hy_ss - (y + 0.5 + offy)) / f_ss * z < yg:
                            continue
                        if z < zb[y, x]:
                            zb[y, x] = z
                            ib[y, x] = i


def build_bands(P, Hs, band_h=32):
    """Bucket primitives by the horizontal bands their bbox touches (for the parallel z-buffer)."""
    kind = P[:, K_KIND]
    ry = np.where(kind == 0, P[:, K_R0] * P[:, K_AY], np.maximum(P[:, K_R0], P[:, K_R1]))
    ymin = np.where(kind == 0, P[:, K_Y0] - ry, np.minimum(P[:, K_Y0], P[:, K_Y1]) - ry) - 2
    ymax = np.where(kind == 0, P[:, K_Y0] + ry, np.maximum(P[:, K_Y0], P[:, K_Y1]) + ry) + 2
    nb = (Hs + band_h - 1) // band_h
    b0 = np.clip((ymin // band_h).astype(np.int64), 0, nb - 1)
    b1 = np.clip((ymax // band_h).astype(np.int64), 0, nb - 1)
    valid = (ymax >= 0) & (ymin < Hs)
    lists = [[] for _ in range(nb)]
    idx = np.nonzero(valid)[0]
    for i in idx:
        for b in range(b0[i], b1[i] + 1):
            lists[b].append(i)
    ptr = np.zeros(nb + 1, np.int64)
    for b in range(nb):
        ptr[b + 1] = ptr[b] + len(lists[b])
    flat = np.zeros(ptr[-1], np.int64)
    for b in range(nb):
        flat[ptr[b]:ptr[b + 1]] = lists[b]
    return flat, ptr


@njit(cache=True, parallel=True, fastmath=True)
def lobe_lines(zb, ib, P, width, rel):
    """Occlusion lines: pixels just behind the silhouette of a nearer primitive (painted lobe separation).
    Returns (H, W) float32 0..1."""
    Hs, Ws = zb.shape
    out = np.zeros((Hs, Ws), np.float32)
    w = int(width)
    for y in prange(Hs):
        for x in range(Ws):
            i = ib[y, x]
            if i < 0:
                continue
            z = zb[y, x]
            best = 0.0
            for dy in range(-w, w + 1):
                yy = y + dy
                if yy < 0 or yy >= Hs:
                    continue
                for dx in range(-w, w + 1):
                    xx = x + dx
                    if xx < 0 or xx >= Ws:
                        continue
                    j = ib[yy, xx]
                    if j < 0 or j == i:
                        continue
                    if P[j, K_MAT] == 2.0 and P[i, K_MAT] == 2.0:
                        continue
                    if zb[yy, xx] < z - rel * z:
                        dd = math.sqrt(dx * dx + dy * dy)
                        v = 1.0 - dd / (w + 1.0)
                        if v > best:
                            best = v
            out[y, x] = best
    return out


@njit(cache=True, inline='always')
def _sphere_n(P, i, x, y):
    cx = P[i, K_X0]
    cy = P[i, K_Y0]
    r = P[i, K_R0]
    ry = r * P[i, K_AY]
    dx = (x - cx) / r
    dy = -(y - cy) / ry
    d2 = dx * dx + dy * dy
    if d2 > 0.999:
        s = 1.0 / math.sqrt(d2 / 0.999)
        dx *= s
        dy *= s
        d2 = 0.999
    return dx, dy, math.sqrt(1.0 - d2)


@njit(cache=True, parallel=True, fastmath=True)
def shade(zb, ib, ao, edge, P, TF, L, pal, bark, sky_fill, bounce, haze, haze_k, haze_max, glow_col, rim_amt, out, offx,
          offy, line_amt, stamen, hy_ss, f_ss, under, fl_own, bedge, bD, bgx, bgy):
    """Paint the z-buffer.  pal: (5,3) deep, shade, mid, lit, hot.  TF[tree] = (cx, cy, rx, ry) form ellipse
    (ss px).  L: (3,) light dir in view coords (x right, y up, z toward viewer)."""
    Hs, Ws = zb.shape
    Lx, Ly, Lz = L[0], L[1], L[2]
    lxy = math.sqrt(Lx * Lx + Ly * Ly) + 1e-6
    for y in prange(Hs):
        for x in range(Ws):
            i = ib[y, x]
            if i < 0:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0
                out[y, x, 2] = 0.0
                out[y, x, 3] = 0.0
                continue
            px = x + 0.5 + offx
            py = y + 0.5 + offy
            mat = int(P[i, K_MAT])
            tr = int(P[i, K_TREE])
            tone = P[i, K_TONE]
            Z = zb[y, x]
            a = ao[y, x]
            if mat == 3:
                # grass blade: dark at the root, lit yellow-green tip
                ax, ay_ = P[i, K_X0], P[i, K_Y0]
                bx, by = P[i, K_X1], P[i, K_Y1]
                ex = bx - ax
                ey = by - ay_
                L2 = ex * ex + ey * ey + 1e-6
                t = min(max(((px - ax) * ex + (py - ay_) * ey) / L2, 0.0), 1.0)
                hv = hash2(i, 7, 97)
                k = sstep(0.2, 0.9, t) * (0.6 + 0.4 * hv)
                r_ = 0.05 + 0.26 * k
                g_ = 0.13 + 0.4 * k
                b_ = 0.12 + 0.12 * k
            elif mat == 2:
                # bark capsule: normal across the capsule
                ax, ay_ = P[i, K_X0], P[i, K_Y0]
                bx, by = P[i, K_X1], P[i, K_Y1]
                ex = bx - ax
                ey = by - ay_
                L2 = ex * ex + ey * ey + 1e-6
                t = ((px - ax) * ex + (py - ay_) * ey) / L2
                t = min(max(t, 0.0), 1.0)
                qx = px - ax - t * ex
                qy = py - ay_ - t * ey
                rr = P[i, K_R0] + (P[i, K_R1] - P[i, K_R0]) * t
                nx = qx / rr
                ny = -qy / rr
                acx = nx
                d2 = min(nx * nx + ny * ny, 0.999)
                nz = math.sqrt(1 - d2)
                # lighting normal from the distance field of the whole wood silhouette: limbs blend smoothly
                # into the trunk (no tube seams where capsules overlap)
                if bD[y, x] >= 0.0:
                    ed_ = 1.0 - min(bD[y, x] / max(rr, 0.5), 1.0)
                    nx = bgx[y, x] * ed_
                    ny = -bgy[y, x] * ed_
                    d2 = min(nx * nx + ny * ny, 0.999)
                    nz = math.sqrt(1 - d2)
                # world arc length along the limb (continuous across chained capsules) and radius (m)
                arc = P[i, K_TONE] + t * P[i, K_AY]
                rw = P[i, K_RZ0] + (P[i, K_RZ1] - P[i, K_RZ0]) * t
                ac = math.asin(max(-1.0, min(1.0, acx))) * rw        # metres around the limb
                # painted two-tone: hard terminator (slightly wobbly, follows the bark relief)
                wob = (fbm2(arc * 3.0, ac * 6.0, 94, 2) - 0.5) * 0.25
                # backlit bark: the sun is behind the trees, so the viewer sees mostly the shadow side
                lam = nx * Lx + ny * Ly + nz * (Lz - 0.35) + wob
                k = sstep(0.2, 0.28, lam)
                r_ = bark[0, 0] + (bark[1, 0] - bark[0, 0]) * k
                g_ = bark[0, 1] + (bark[1, 1] - bark[0, 1]) * k
                b_ = bark[0, 2] + (bark[1, 2] - bark[0, 2]) * k
                # soft modelling inside the lit side
                r_ *= 0.92 + 0.14 * sstep(0.3, 0.9, lam) * k
                g_ *= 0.92 + 0.14 * sstep(0.3, 0.9, lam) * k
                b_ *= 0.92 + 0.14 * sstep(0.3, 0.9, lam) * k
                # rough bark: vertical fissure streaks + horizontal lenticel bands (typical of cherry)
                nz_ = fbm2(arc * 2.2, ac * 9.0, 91, 3)
                bpos = arc / 0.07
                bi_ = math.floor(bpos)
                bf = bpos - bi_
                hb_ = hash2(int(bi_), 3, 92)
                seg_ = vnoise(ac * 7.0 + hb_ * 13.0, bi_ * 0.7, 95)          # bands broken around the limb
                bw = 0.12 + 0.18 * hb_
                band = (1.0 - sstep(bw * 0.4, bw * 0.6, abs(bf - 0.5))) * sstep(0.3, 0.42, seg_) * (hb_ > 0.25)
                pixm = rw * 2.0 / max(P[i, K_R0] + (P[i, K_R1] - P[i, K_R0]) * t, 0.5)   # metres per ss px
                band *= 1.0 - sstep(0.012, 0.03, pixm)
                lip = (1.0 - sstep(0.0, bw, abs(bf - 0.5 - bw))) * sstep(0.35, 0.5, seg_) * (hb_ > 0.3) * \
                    (1.0 - sstep(0.012, 0.03, pixm))
                tv = 0.86 + 0.28 * nz_ - 0.62 * band + 0.35 * lip * k
                # silvery horizontal lenticels: short light dashes wrapping around the limb
                dtl = 1.0 - sstep(0.004, 0.012, pixm)
                lpos = arc / 0.045
                li_ = math.floor(lpos)
                lf_ = lpos - li_
                lh_ = hash2(int(li_), 7, 96)
                lseg = sstep(0.55, 0.62, vnoise(ac * 11.0 + lh_ * 29.0, li_ * 1.3, 97))
                lent = (1.0 - sstep(0.05, 0.1 + 0.08 * lh_, abs(lf_ - 0.5))) * lseg * (lh_ > 0.35) * dtl
                # knots: dark oval scars with a lit lower lip
                kpos = arc / 0.55
                ki_ = math.floor(kpos)
                kh_ = hash2(int(ki_), 11, 98)
                kc = (kpos - ki_ - 0.5) * 0.55 / 0.07
                kac = (ac - (kh_ - 0.5) * rw * 1.2) / (0.05 + 0.03 * kh_)
                kd = math.sqrt(kc * kc + kac * kac)
                knot = (1.0 - sstep(0.7, 1.0, kd)) * (kh_ > 0.55) * dtl
                klip = (1.0 - sstep(0.0, 0.35, abs(kd - 1.05))) * (kc > 0.0) * (kh_ > 0.55) * dtl
                tv += 0.9 * lent * (0.5 + 0.5 * k) - 0.55 * knot + 0.5 * klip * (0.4 + 0.6 * k)
                tv -= 0.25 * band * (1.0 - k)
                lich = sstep(0.58, 0.75, fbm2(arc * 1.3 + 3.0, ac * 3.0, 93, 3)) * 0.35
                r_ *= tv
                g_ *= tv
                b_ *= tv
                # silvery-grey lenticel dashes and band lips read even on the dark shadow side of the bark
                r_ += 0.2 * lent * (1.0 - k) + 0.08 * lip * (1.0 - k)
                g_ += 0.18 * lent * (1.0 - k) + 0.07 * lip * (1.0 - k)
                b_ += 0.23 * lent * (1.0 - k) + 0.09 * lip * (1.0 - k)
                lv = 0.16 + 0.3 * k
                r_ += (lv * 0.95 - r_) * lich
                g_ += (lv * 1.0 - g_) * lich
                b_ += (lv * 0.98 - b_) * lich
                sh = 1.0 - 0.45 * a - 0.5 * line_amt * edge[y, x]
                r_ *= sh
                g_ *= sh
                b_ *= sh
                ed = math.sqrt(d2)
                # bright warm rim where the sun grazes the silhouette
                # warm rim only on the OUTER silhouette of the wood facing the sun (no seams between capsules)
                rim = bedge[y, x] * (1.0 - 0.4 * band)
                r_ += (1.08 - r_) * rim * 0.85 * rim_amt
                g_ += (0.76 - g_) * rim * 0.85 * rim_amt
                b_ += (0.5 - b_) * rim * 0.85 * rim_amt
                # cool sky / bounce light rim on the shadow side
                cb = sstep(0.62, 0.9, ed) * sstep(0.05, 0.5, -(nx * Lx) / lxy) * (1.0 - k) * (1.0 - 0.6 * a)
                r_ += (0.3 - r_) * cb * 0.6
                g_ += (0.33 - g_) * cb * 0.6
                b_ += (0.6 - b_) * cb * 0.6
            else:
                ox, oy, oz = _sphere_n(P, i, px, py)
                par = int(P[i, K_PAR])
                if par >= 0:
                    qx, qy, qz = _sphere_n(P, par, px, py)
                else:
                    qx, qy, qz = ox, oy, oz
                # clump-level normal (the core dome of the whole clump): big masses carry the light,
                # the small lobes only break the silhouettes (no 'cotton ball' highlight on every lobe)
                gpar = par
                if par >= 0 and int(P[par, K_PAR]) >= 0:
                    gpar = int(P[par, K_PAR])
                if gpar >= 0 and gpar != par:
                    gx_, gy_, gz_ = _sphere_n(P, gpar, px, py)
                else:
                    gx_, gy_, gz_ = qx, qy, qz
                fcx, fcy, frx, fry = TF[tr, 0], TF[tr, 1], TF[tr, 2], TF[tr, 3]
                fx = (px - fcx) / frx
                fy = -(py - fcy) / fry
                fd = fx * fx + fy * fy
                if fd > 0.95:
                    s = 1.0 / math.sqrt(fd / 0.95)
                    fx *= s
                    fy *= s
                    fd = 0.95
                fz = math.sqrt(1 - fd)
                el = (hy_ss - py) / f_ss
                if mat == 1:
                    # florets only keep their own modelling on the lit side; in shade they melt into the mass
                    n0x = 0.28 * qx + 0.72 * fx
                    n0y = 0.28 * qy + 0.72 * fy
                    n0z = 0.28 * qz + 0.72 * fz
                    if el > 0.0:
                        n0y -= under * min(el * 2.0, 1.0)
                    n0 = math.sqrt(n0x * n0x + n0y * n0y + n0z * n0z) + 1e-6
                    v0 = 0.47 + 0.78 * (n0x * Lx + n0y * Ly + n0z * Lz) / n0 - 0.45 * a
                    litk = sstep(0.4, 0.68, v0)
                    fo = fl_own * (0.25 + 0.75 * litk)
                    w1, w2, w3 = fo + 0.04, 0.3, 0.66 - fo
                    qx = 0.55 * qx + 0.45 * gx_
                    qy = 0.55 * qy + 0.45 * gy_
                    qz = 0.55 * qz + 0.45 * gz_
                    sil = 1.0 - sstep(0.35, 0.7, qz)          # 1 at the rim of the parent dome
                    lw = (0.25 + 0.35 * litk) * (0.2 + 0.8 * sil)
                    if par >= 0:
                        tone = P[par, K_TONE]                 # no per-floret value speckle inside the mass
                else:
                    w1, w2, w3 = 0.22, 0.28, 0.5
                    lw = 1.0
                    sil = 1.0
                    if par >= 0:
                        # child lobes: weak own modelling -> big lobe shapes read as one mass
                        w1, w2, w3 = 0.08, 0.4, 0.52
                        lw = 0.7
                nx = w1 * ox + w2 * qx + w3 * fx
                ny = w1 * oy + w2 * qy + w3 * fy
                nz = w1 * oz + w2 * qz + w3 * fz
                # masses above the eye are seen from below: we look at their shaded undersides
                if el > 0.0:
                    ny -= under * min(el * 2.0, 1.0)
                nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
                nx /= nn
                ny /= nn
                nz /= nn
                lam = nx * Lx + ny * Ly + nz * Lz
                v = 0.47 + 0.78 * lam + tone * 0.05 - 0.45 * a - line_amt * edge[y, x] * lw
                ti = i if (mat == 0 or par < 0) else par
                ftc, fts = flower_tex(px - P[ti, K_X0], py - P[ti, K_Y0], P[ti, K_Z0] / f_ss, float(ti % 997))
                v += ftc * 0.025
                # painterly value steps: soft posterisation
                v = min(max(v, 0.0), 1.0)
                q = v * 4.0
                qi = math.floor(q)
                qf = q - qi
                qf = sstep(0.44, 0.56, qf)
                v = (qi + qf) / 4.0
                v = min(max(v, 0.0), 0.999)
                seg = v * 4.0
                si = int(seg)
                sf = seg - si
                r_ = pal[si, 0] + (pal[si + 1, 0] - pal[si, 0]) * sf
                g_ = pal[si, 1] + (pal[si + 1, 1] - pal[si, 1]) * sf
                b_ = pal[si, 2] + (pal[si + 1, 2] - pal[si, 2]) * sf
                # sky fill on upward-facing shadow, warm bounce on downward-facing shadow
                shd = 1.0 - sstep(0.25, 0.6, v)
                up = max(ny, 0.0) * shd * 0.3
                dn = max(-ny, 0.0) * shd * 0.25
                r_ += (sky_fill[0] - r_) * up + (bounce[0] - r_) * dn
                g_ += (sky_fill[1] - g_) * up + (bounce[1] - g_) * dn
                b_ += (sky_fill[2] - b_) * up + (bounce[2] - b_) * dn
                # individual flowers: bright white-pink in the light, pale lilac-pink specks in the shade
                if fts > 0.0:
                    lt = sstep(0.35, 0.75, v)
                    tr_ = pal[2, 0] * (1 - lt) + pal[4, 0] * lt
                    tg_ = pal[2, 1] * 0.92 * (1 - lt) + pal[4, 1] * lt
                    tb_ = pal[2, 2] * (1 - lt) + pal[4, 2] * lt
                    sa = fts * (0.08 + 0.5 * lt) * (1.0 - 0.7 * a)
                    r_ += (tr_ - r_) * sa
                    g_ += (tg_ - g_) * sa
                    b_ += (tb_ - b_) * sa
                # individually lit little flowers inside the lit masses + pink stamen centres up close
                if mat == 1:
                    lo = ox * Lx + oy * Ly + oz * Lz
                    hsz = hash2(i, 3, 211)
                    hl = sstep(0.5 + 0.25 * hsz, 0.56 + 0.25 * hsz, lo) * sstep(0.6, 0.78, v) * (0.25 + 0.4 * hsz) * (hash2(i, 5, 212) > 0.45) * (1.0 - sstep(0.4, 0.75, qz))
                    r_ += (pal[4, 0] - r_) * hl
                    g_ += (pal[4, 1] - g_) * hl
                    b_ += (pal[4, 2] - b_) * hl
                    rp = P[i, K_R0]
                    if stamen > 0.0 and rp > 5.0:
                        dc = ox * ox + oy * oy
                        st = (1.0 - sstep(0.02, 0.07, dc)) * stamen * sstep(14.0, 22.0, rp)
                        r_ += (0.93 - r_) * st
                        g_ += (0.42 - g_) * st
                        b_ += (0.58 - b_) * st
                # translucent glow on thin sun-facing edges (petals are translucent when backlit)
                if mat == 1:
                    # florets glow only where they form the rim of their dome (no crescent on every floret)
                    thin = (1.0 - qz) * (1.0 - qz) * (0.3 + 0.7 * sil)
                    facing = max(0.0, (qx * Lx + qy * Ly) / lxy)
                else:
                    thin = (1.0 - oz) * (1.0 - oz)
                    facing = max(0.0, (ox * Lx + oy * Ly) / lxy)
                gl = thin * facing * rim_amt * (1.0 - 0.6 * a)
                r_ += glow_col[0] * gl
                g_ += glow_col[1] * gl
                b_ += glow_col[2] * gl
            # aerial perspective
            h = haze_max * (1.0 - math.exp(-Z / haze_k))
            r_ += (haze[0] - r_) * h
            g_ += (haze[1] - g_) * h
            b_ += (haze[2] - b_) * h
            out[y, x, 0] = r_
            out[y, x, 1] = g_
            out[y, x, 2] = b_
            out[y, x, 3] = 1.0


# ----------------------------------------------------------------------------- petals

@njit(cache=True, fastmath=True)
def draw_petals(img, xs, ys, ax, bx, ang, coc, col, alpha):
    """Composite soft tumbling petals (far -> near order expected) onto img (H, W, 3) in place.
    ax/bx: semi-axes px; ang: rotation; coc: blur radius px; col: (n, 3); alpha: (n,) opacity."""
    H, W = img.shape[0], img.shape[1]
    n = xs.shape[0]
    for i in range(n):
        a = ax[i]
        b = bx[i]
        c = coc[i]
        R = a + c + 2.0
        x0 = max(0, int(xs[i] - R))
        x1 = min(W, int(xs[i] + R + 2))
        y0 = max(0, int(ys[i] - R))
        y1 = min(H, int(ys[i] + R + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        ca = math.cos(ang[i])
        sa = math.sin(ang[i])
        soft = 0.7 + c
        # energy conservation for defocused petals
        k = (a * b) / ((a + 0.5 * c) * (b + 0.5 * c))
        op = alpha[i] * min(1.0, k * 1.15)
        cr, cg, cb = col[i, 0], col[i, 1], col[i, 2]
        for y in range(y0, y1):
            dy = y + 0.5 - ys[i]
            for x in range(x0, x1):
                dx = x + 0.5 - xs[i]
                u = dx * ca + dy * sa
                v = -dx * sa + dy * ca
                # notched petal: ellipse, slightly egg-shaped, with a small notch at +u tip
                au = a * (1.0 + 0.12 * (u / (a + 1e-3)))
                q = math.sqrt((u / (au + 1e-3)) ** 2 + (v / (b + 1e-3)) ** 2)
                dist = (q - 1.0) * min(a, b)
                if c < 1.5:
                    nt = math.exp(-(v / (0.25 * b + 0.3)) ** 2) * sstep(0.55 * a, a, u) * 0.35 * min(a, b)
                    dist += nt
                al = 1.0 - sstep(-soft, soft, dist)
                if al <= 0.002:
                    continue
                al *= op
                # soft inner gradient: base of petal slightly deeper pink
                g = sstep(-a, a, -u) * 0.12
                img[y, x, 0] = img[y, x, 0] * (1 - al) + (cr - g * 0.05) * al
                img[y, x, 1] = img[y, x, 1] * (1 - al) + (cg - g * 0.35) * al
                img[y, x, 2] = img[y, x, 2] * (1 - al) + (cb - g * 0.2) * al
