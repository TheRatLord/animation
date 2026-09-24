"""Painted cloud-sea for s10: rows of cauliflower billows at true perspective depths.

World: camera at (camx, camy, 0) looking along +z; the cloud deck sits at height 0.  The deck is a
stack of ROWS at fixed world depths Z_i (geometric spacing).  Each row is a periodic strip of 3D
ellipsoidal lobes (big billows -> medium lobes on their crowns -> small beads on the silhouettes) plus
a solid body below them, so every row is fully OPAQUE.  One numba kernel walks the rows front to back
for every sample and takes the first row whose surface covers it (hard z-buffer inside a row -> crisp
internal lobe contours), then paints it:

  * flattened cel ramp: deep violet-blue valley / lavender shadow / coral midtone / peach-gold lit /
    warm-white hot cap - soft gradient only inside each band, narrow transitions between bands;
  * light comes from the low sun ahead (backlight): crowns and sun-facing flanks are lit, faces
    toward the camera fall into cool shadow, valleys deepen with depth below the crowns;
  * a crisp 1-2 px gold rim on every sun-facing edge (tested analytically: the sample one rim-width
    toward the sun is empty or belongs to a farther lobe) + a softer translucent glow inside it;
  * a glittering glory path: lit tones strongest along the sun axis, falling to pink/violet flanks;
  * aerial perspective toward a luminous gold haze at the horizon.
Secondary motion: lobes boil slowly (radius breathing) and each row drifts with the wind.
"""
import math
import numpy as np
from numba import njit, prange

# lobe columns
LX, LH, LR, LAY, LDZ, LLEV, LPH, LPAR = range(8)
NLC = 8


def _hx(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# palette rows (see _shade): 0 deep valley, 1 shadow, 2 shadow up-facing (sky fill), 3 coral mid,
# 4 lit (sun axis, gold), 5 lit (flanks, pink), 6 hot cap, 7 rim, 8 haze sun, 9 haze away, 10 bounce
PALETTE = np.array([_hx(c) for c in (
    '#18165a', '#3a3292', '#5e56b0', '#e2789c', '#ffb27a', '#f1909e', '#ffe2b0', '#fff0c8',
    '#ffe4b2', '#cf96c4', '#8a5aa8')], np.float32)
PALETTE[7] *= 1.7          # HDR rim so bloom picks it up
PALETTE[6] *= 1.08


def build_rows(seed=5, z0=1.2, z1=700.0, ratio=1.085, s0=0.42):
    """Returns dict of row arrays + lobe table + per-row bins (CSR)."""
    rng = np.random.default_rng(seed)
    Rz, Rs, Rp, Rmax, Rdrift = [], [], [], [], []
    lobes = []
    rstart = [0]
    z = z0
    ri = 0
    while z < z1:
        s = s0 * max(z / 5.0, 1.0) ** 0.5
        s = min(s, 3.2)
        P = 70.0 * s
        rows_lobes = []
        # big-scale envelope along the row: mounds and troughs (breaks the row rhythm)
        k1, k2 = rng.uniform(1.0, 2.5) * 2 * math.pi / P * 3, rng.uniform(3.0, 6.0) * 2 * math.pi / P * 3
        p1, p2 = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
        near = z < 14.0
        x = 0.0
        while x < P:
            env = 0.55 + 0.3 * math.sin(k1 * x + p1) + 0.15 * math.sin(k2 * x + p2)
            r = s * rng.uniform(0.7, 1.25) * (0.75 + 0.5 * env)
            u = rng.random()
            if u < 0.12:
                r *= 1.7
            if near and u > 0.955:
                r *= 2.6          # hero masses close to camera
            ay = rng.uniform(0.62, 0.8)
            hc = (rng.uniform(-0.35, 0.05) + 0.35 * (env - 0.5)) * r
            dz = rng.uniform(-0.25, 0.25) * r
            base_i = len(rows_lobes)
            rows_lobes.append([x, hc, r, ay, dz, 0, rng.uniform(0, 6.3), -1])
            # level 1: medium lobes on the crown (upper hemisphere, some facing the camera)
            nch = int(rng.integers(5, 9) * max(r / s, 1.0) ** 1.2)
            for _ in range(nch):
                th = rng.uniform(0.18, math.pi - 0.18)
                ez = rng.uniform(0.0, 0.3)
                sxy = math.sqrt(max(1 - ez * ez, 0.0))
                rc = r * rng.uniform(0.24, 0.46)
                d = r - rc * rng.uniform(0.3, 0.6)
                cx = x + math.cos(th) * sxy * d
                ch = hc + math.sin(th) * sxy * d * ay
                cdz = dz + ez * d * 0.8
                ci = len(rows_lobes)
                rows_lobes.append([cx, ch, rc, rng.uniform(0.7, 0.9), cdz, 1, rng.uniform(0, 6.3), base_i])
                # level 2: small beads on the silhouette / crown of the child
                for _ in range(int(rng.integers(2, 5))):
                    th2 = rng.uniform(0.25, math.pi - 0.25)
                    ez2 = rng.uniform(0.0, 0.35)
                    s2 = math.sqrt(max(1 - ez2 * ez2, 0.0))
                    rb = rc * rng.uniform(0.28, 0.45)
                    d2 = rc - rb * rng.uniform(0.25, 0.55)
                    rows_lobes.append([cx + math.cos(th2) * s2 * d2, ch + math.sin(th2) * s2 * d2 * 0.85, rb,
                                       rng.uniform(0.75, 0.92), cdz + ez2 * d2 * 0.8, 2, rng.uniform(0, 6.3), ci])
            x += r * rng.uniform(0.95, 1.45)
        arr = np.array(rows_lobes, np.float64)
        arr[:, LX] = np.mod(arr[:, LX], P)
        arr[arr[:, LPAR] >= 0, LPAR] += rstart[-1]
        top = float(np.max(arr[:, LH] + arr[:, LR] * arr[:, LAY])) * 1.06
        lobes.append(arr)
        Rz.append(z)
        Rs.append(s)
        Rp.append(P)
        Rmax.append(top)
        Rdrift.append(rng.uniform(0.02, 0.06) * s * (1 if rng.random() < 0.7 else -0.6))
        rstart.append(rstart[-1] + len(arr))
        z *= ratio
        ri += 1
    L = np.concatenate(lobes, 0)
    # bins per row (width = s), CSR
    binoff = [0]
    binptr = []
    binidx = []
    nbins = []
    for i in range(len(Rz)):
        s, P = Rs[i], Rp[i]
        nb = max(int(P / s), 1)
        bw = P / nb
        buckets = [[] for _ in range(nb)]
        for j in range(rstart[i], rstart[i + 1]):
            x, r = L[j, LX], L[j, LR] * 1.08
            b0 = int(math.floor((x - r) / bw))
            b1 = int(math.floor((x + r) / bw))
            for b in range(b0, b1 + 1):
                buckets[b % nb].append(j)
        nbins.append(nb)
        for b in range(nb):
            binptr.append(len(binidx))
            binidx.extend(buckets[b])
        binoff.append(len(binptr))
    binptr.append(len(binidx))
    return dict(Rz=np.array(Rz), Rs=np.array(Rs), Rp=np.array(Rp), Rmax=np.array(Rmax), Rdrift=np.array(Rdrift),
                L=L, binoff=np.array(binoff, np.int64), binptr=np.array(binptr, np.int64),
                binidx=np.array(binidx, np.int64), nbins=np.array(nbins, np.int64))


@njit(cache=True, fastmath=True, inline='always')
def _cover(L, Lr, binptr, binidx, bo, nb, bw, P, xm, h, wamp, s):
    """Front-most lobe of one row covering (xm, h). returns (index, depth, ex, ey, ez)."""
    b = int(xm / bw)
    if b >= nb:
        b = nb - 1
    if b < 0:
        b = 0
    best = -1e18
    bi = -1
    bex = 0.0
    bey = 0.0
    bez = 1.0
    k0 = binptr[bo + b]
    k1 = binptr[bo + b + 1]
    for k in range(k0, k1):
        j = binidx[k]
        r = Lr[j]
        dx = xm - L[j, 0]
        if dx > 0.5 * P:
            dx -= P
        elif dx < -0.5 * P:
            dx += P
        if dx > 1.1 * r or dx < -1.1 * r:
            continue
        ry = r * L[j, 3]
        dh = h - L[j, 1]
        if dh > 1.1 * ry or dh < -1.1 * ry:
            continue
        if L[j, 4] + r < best:
            continue
        ph = L[j, 6]
        # painterly wobble of the lobe outline (no clean circles)
        dx2 = dx + wamp * s * math.sin(dh / s * 5.1 + ph * 3.0)
        dh2 = dh + wamp * s * 0.7 * math.sin(dx / s * 4.3 + ph * 2.0)
        ex = dx2 / r
        ey = dh2 / ry
        q = ex * ex + ey * ey
        if q >= 1.0:
            continue
        ez = math.sqrt(1.0 - q)
        dep = L[j, 4] + r * ez
        if dep > best:
            best = dep
            bi = j
            bex = ex
            bey = ey
            bez = ez
    return bi, best, bex, bey, bez


@njit(cache=True, fastmath=True, inline='always')
def _pnormal(L, Lr, j, xm, h, P):
    """Normal of lobe j's ellipsoid at the point (xm, h) (clamped inside)."""
    r = Lr[j]
    dx = xm - L[j, 0]
    if dx > 0.5 * P:
        dx -= P
    elif dx < -0.5 * P:
        dx += P
    ex = dx / r
    ey = (h - L[j, 1]) / (r * L[j, 3])
    q = ex * ex + ey * ey
    if q > 0.97:
        k = math.sqrt(0.97 / q)
        ex *= k
        ey *= k
        q = 0.97
    ez = math.sqrt(1.0 - q)
    nx = ex
    ny = ey / L[j, 3]
    nz = ez
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / n, ny / n, nz / n


@njit(cache=True, fastmath=True, parallel=True)
def render(Ws, Hs, f, hy, cx0, camx, camy, d, t, sx, sy, W0, fog_z, gp_w, light, rim_px,
           Rz, Rs, Rp, Rmax, Rdrift, L, Lr, binoff, binptr, binidx, nbins, pal, zmin_row):
    """Returns rgb (Hs, Ws, 3), alpha (Hs, Ws), zc (Hs, Ws; camera depth, 0 = sky), lit (Hs, Ws)."""
    rgb = np.zeros((Hs, Ws, 3), np.float32)
    al = np.zeros((Hs, Ws), np.float32)
    zc = np.zeros((Hs, Ws), np.float32)
    litb = np.zeros((Hs, Ws), np.float32)
    nrow = Rz.shape[0]
    wamp = 0.045
    for py in prange(Hs):
        yy = py + 0.5
        for px in range(Ws):
            xx = px + 0.5
            hit = -1
            jhit = -1
            nx = 0.0
            ny = 0.0
            nz = 1.0
            h = 0.0
            s = 1.0
            dep = 0.0
            xm = 0.0
            for i in range(nrow):
                z = Rz[i] - d
                if z < zmin_row:
                    continue
                h = camy - (yy - hy) * z / f
                if h > Rmax[i]:
                    continue
                s = Rs[i]
                P = Rp[i]
                wx = camx + (xx - cx0) * z / f - Rdrift[i] * t
                xm = wx - math.floor(wx / P) * P
                nb = nbins[i]
                bw = P / nb
                j, dp, ex, ey, ez = _cover(L, Lr, binptr, binidx, binoff[i], nb, bw, P, xm, h, wamp, s)
                if j >= 0:
                    hit = i
                    jhit = j
                    dep = dp
                    ay = L[j, 3]
                    nx = ex
                    ny = ey / ay
                    nz = ez
                    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                    nx /= nn
                    ny /= nn
                    nz /= nn
                    # blend with the parent chain so each billow keeps one coherent light/shadow shape
                    pj = int(L[j, 7])
                    w = 0.45
                    while pj >= 0:
                        qx, qy, qz = _pnormal(L, Lr, pj, xm, h, P)
                        nx = w * nx + (1 - w) * qx
                        ny = w * ny + (1 - w) * qy
                        nz = w * nz + (1 - w) * qz
                        pj = int(L[pj, 7])
                        w = 0.6
                    nx = 0.85 * nx
                    ny = 0.85 * ny
                    nz = 0.85 * nz + 0.15
                    break
                if h < -0.3 * s:
                    hit = i
                    nx = 0.0
                    ny = -0.1
                    nz = 0.99
                    dep = -0.3 * s
                    break
            if hit < 0:
                continue
            i = hit
            z = Rz[i] - d
            # ---------------- light (stylised backlight from the low sun ahead)
            lx = (sx - xx) / W0 * 1.8
            if lx > 0.9:
                lx = 0.9
            elif lx < -0.9:
                lx = -0.9
            ly = 0.34
            lz = -0.62
            ln = math.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= ln
            ly /= ln
            lz /= ln
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
            nx /= nn
            ny /= nn
            nz /= nn
            ndl = nx * lx + ny * ly + nz * lz
            hrel = h / s
            lit_ok0 = (hrel + 0.4) / 0.6
            lit_ok0 = 0.0 if lit_ok0 < 0 else (1.0 if lit_ok0 > 1 else lit_ok0)
            # glory path toward the sun (widens toward the camera)
            yn = (yy - hy) / (Hs - hy)
            if yn < 0.0:
                yn = 0.0
            sig = W0 * (gp_w + 0.16 * yn)
            gx = (xx - sx) / sig
            gp = math.exp(-gx * gx)
            gp2 = math.exp(-gx * gx * 0.2)
            # ---------------- shadow: deep valley -> lavender with height; up-facing catches sky fill
            vh = (hrel + 0.5) / 1.9
            if vh < 0.0:
                vh = 0.0
            elif vh > 1.0:
                vh = 1.0
            vh = vh * vh * (3 - 2 * vh)
            c0 = pal[0, 0] + (pal[1, 0] - pal[0, 0]) * vh
            c1 = pal[0, 1] + (pal[1, 1] - pal[0, 1]) * vh
            c2 = pal[0, 2] + (pal[1, 2] - pal[0, 2]) * vh
            up = ny
            if up < 0.0:
                up = 0.0
            fu = up * 0.6 * vh
            c0 += (pal[2, 0] - c0) * fu
            c1 += (pal[2, 1] - c1) * fu
            c2 += (pal[2, 2] - c2) * fu
            # shadow side: soft internal gradient, lighter / pinker toward the terminator (translucency)
            tg = (ndl + 0.45) / 0.45
            tg = 0.0 if tg < 0 else (1.0 if tg > 1 else tg)
            tg = tg * tg * 0.3 * lit_ok0
            c0 += (pal[2, 0] * 1.1 + 0.08 - c0) * tg
            c1 += (pal[2, 1] - c1) * tg
            c2 += (pal[2, 2] - c2) * tg
            dn = -ny
            if dn > 0.0:
                fb = dn * 0.4 * gp2 * (1 - vh * 0.5)
                c0 += (pal[10, 0] - c0) * fb
                c1 += (pal[10, 1] - c1) * fb
                c2 += (pal[10, 2] - c2) * fb
            # lit colour: gold on the sun axis, pink on the flanks
            lr = pal[5, 0] + (pal[4, 0] - pal[5, 0]) * gp2
            lg = pal[5, 1] + (pal[4, 1] - pal[5, 1]) * gp2
            lb = pal[5, 2] + (pal[4, 2] - pal[5, 2]) * gp2
            ints = 0.84 + 0.22 * gp2 + 0.22 * gp + 0.1 * light
            lr *= ints
            lg *= ints
            lb *= ints
            lit_ok = (hrel + 0.1) / 0.45
            if lit_ok < 0.0:
                lit_ok = 0.0
            elif lit_ok > 1.0:
                lit_ok = 1.0
            v = ndl
            tm = (v + 0.03) / 0.04
            tm = 0.0 if tm < 0 else (1.0 if tm > 1 else tm)
            tl = (v - 0.05) / 0.04
            tl = 0.0 if tl < 0 else (1.0 if tl > 1 else tl)
            th = (v - 0.48) / 0.07
            th = 0.0 if th < 0 else (1.0 if th > 1 else th)
            tm *= lit_ok
            tl *= lit_ok
            th *= lit_ok * (0.3 + 0.7 * gp2)
            mr = pal[3, 0] * (0.85 + 0.2 * gp2)
            mg = pal[3, 1] * (0.85 + 0.25 * gp2)
            mb = pal[3, 2] * (0.9 + 0.1 * gp2)
            c0 += (mr - c0) * tm * 0.8
            c1 += (mg - c1) * tm * 0.8
            c2 += (mb - c2) * tm * 0.8
            ig = 0.82 + 0.3 * min(max((v - 0.05) / 0.4, 0.0), 1.0)
            c0 += (lr * ig - c0) * tl
            c1 += (lg * ig - c1) * tl
            c2 += (lb * ig - c2) * tl
            hc = 0.8 + 0.3 * gp2
            c0 += (pal[6, 0] * hc - c0) * th
            c1 += (pal[6, 1] * hc - c1) * th
            c2 += (pal[6, 2] * hc - c2) * th
            # ---------------- rim: is the sample one rim-width toward the sun empty / farther?
            if jhit >= 0:
                P = Rp[i]
                nb = nbins[i]
                bw = P / nb
                dxs = sx - xx
                dys = sy - yy
                dl = math.sqrt(dxs * dxs + dys * dys) + 1e-3
                ux = 0.6 * dxs / dl
                uy = -1.0
                un = math.sqrt(ux * ux + uy * uy)
                ux /= un
                uy /= un
                pxw = z / f
                rim = 0.0
                glow = 0.0
                for step in range(2):
                    off = (rim_px if step == 0 else rim_px * 3.5) * pxw
                    xo = xm + ux * off
                    xo = xo - math.floor(xo / P) * P
                    ho = h - uy * off
                    j2, dp2, a1, a2, a3 = _cover(L, Lr, binptr, binidx, binoff[i], nb, bw, P, xo, ho, wamp, Rs[i])
                    e = 0.0
                    if j2 < 0:
                        e = 1.0 if ho > -0.3 * Rs[i] else 0.0
                    elif dep - dp2 > 0.15 * Lr[jhit]:
                        e = 0.7
                    if step == 0:
                        rim = e
                    else:
                        glow = e
                rw = (0.3 + 0.55 * gp2 + 0.4 * gp) * lit_ok * (0.45 + 0.55 * tm)
                gw = glow * 0.3 * rw
                c0 += pal[7, 0] * 0.5 * gw
                c1 += pal[7, 1] * 0.33 * gw
                c2 += pal[7, 2] * 0.2 * gw
                rr = rim * rw
                if rr > 1.0:
                    rr = 1.0
                k = 0.8 + 0.4 * light
                c0 += (pal[7, 0] * k - c0) * rr
                c1 += (pal[7, 1] * k - c1) * rr
                c2 += (pal[7, 2] * k - c2) * rr
                tl = max(tl, rr)
            # ---------------- aerial perspective toward the gold horizon haze
            fg = 1.0 - math.exp(-((z / fog_z) ** 1.2))
            hzr = pal[9, 0] + (pal[8, 0] - pal[9, 0]) * gp2
            hzg = pal[9, 1] + (pal[8, 1] - pal[9, 1]) * gp2
            hzb = pal[9, 2] + (pal[8, 2] - pal[9, 2]) * gp2
            c0 += (hzr - c0) * fg
            c1 += (hzg - c1) * fg
            c2 += (hzb - c2) * fg
            rgb[py, px, 0] = c0
            rgb[py, px, 1] = c1
            rgb[py, px, 2] = c2
            al[py, px] = 1.0
            zc[py, px] = z - dep
            litb[py, px] = tl * (1 - fg)
    return rgb, al, zc, litb


def radii(R, t):
    L = R['L']
    return L[:, LR] * (1.0 + 0.035 * np.sin(0.7 * t + L[:, LPH]))


def draw(R, Ws, Hs, f, hy, cx0, camx, camy, d, t, sx, sy, W0, fog_z=150.0, gp_w=0.05, light=0.0, rim_px=2.2,
         pal=None, zmin_row=0.5):
    return render(Ws, Hs, f, hy, cx0, camx, camy, d, t, sx, sy, W0, fog_z, gp_w, light, rim_px,
                  R['Rz'], R['Rs'], R['Rp'], R['Rmax'], R['Rdrift'], R['L'], radii(R, t), R['binoff'],
                  R['binptr'], R['binidx'], R['nbins'], PALETTE if pal is None else pal, zmin_row)


