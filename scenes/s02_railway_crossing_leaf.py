"""Painted broadleaf foliage for s02_railway_crossing (round 3).

A crown is painted the way a Shinkai background artist paints summer trees: MANY small leaf masses
(dozens per crown) laid back to front, each with a SERRATED silhouette (pointed leaf tips, rounded
notches between them), and a strict 3-value scheme - cool teal shadow, mid green, warm yellow-green light -
chosen from the CROWN's form (one ellipsoid per tree, so each tree reads as one lit volume with a clean
terminator) nudged by each mass's own top/bottom. Lower masses sit in front of the ones above them, so
each lit, toothed top edge is cut against the shadowed underside behind it; nearer masses cast a short
shadow down-left; a thin sun-side rim lights the outer silhouette; a few sky holes near the edge.

Also returns a sway-weight map (0 at the crown base / trunks .. 1 at the crown top) so the scene can
apply a subtle, temporally smooth wind sway per frame.
"""
import math

import numpy as np
import cv2
from numba import njit

import s02_railway_crossing_paint as P

hx = P.hexc
sstep = P.sstep

# clump columns
(C_X, C_Y, C_R, C_AS, C_Z, C_GX, C_GY, C_GRX, C_GRY, C_TONE, C_FOG, C_N, C_PH, C_AMP, C_L1, C_L2, C_FLAT,
 C_TREE, C_BIAS) = range(19)
NC = 19

PAL = dict(gap=hx('#0a2a30'), core=hx('#123c46'), shd=hx('#1f5a56'), shd_w=hx('#2e6c52'), mid=hx('#4a943a'),
           mid_w=hx('#64a638'), lit=hx('#a6d24a'), hi=hx('#e4f28a'), rim=hx('#f4fcc0'), haze=hx('#a9cde2'),
           trunk=hx('#3a3434'))


@njit(cache=True, fastmath=True)
def _zbuf(Hs, Ws, S, ss):
    n = S.shape[0]
    hb = np.full((Hs, Ws), -1e12, np.float32)
    ib = np.full((Hs, Ws), -1, np.int32)
    for i in range(n):
        cx, cy = S[i, 0] * ss, S[i, 1] * ss
        r = S[i, 2] * ss
        ry = r * S[i, 3]
        z0 = S[i, 4] * ss
        nt, ph, amp = S[i, 11], S[i, 12], S[i, 13]
        l1, l2, flat = S[i, 14], S[i, 15], S[i, 16]
        R = r * 1.35 + 2.0
        x0 = max(0, int(cx - R))
        x1 = min(Ws, int(cx + R) + 2)
        y0 = max(0, int(cy - R))
        y1 = min(Hs, int(cy + R) + 2)
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / ry
            if dy > 0:
                dy = dy * (1.0 + flat)
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.3:
                    continue
                th = math.atan2(dy, dx)
                # low-frequency lumpiness of the mass
                lump = 1.0 + 0.12 * math.sin(2.0 * th + l1) + 0.08 * math.sin(3.0 * th + l2)
                # serrated leaf teeth: a zigzag of pointed leaf tips with slightly convex flanks and uneven
                # tooth sizes (irregular spacing via a warp of theta); the underside stays smoother
                tw = th + 0.35 * math.sin(3.0 * th + l2 * 1.7)
                tf = tw / 6.2831853 * nt + ph
                ti = math.floor(tf)
                u = tf - ti
                q = abs(2.0 * u - 1.0)
                hsh = math.sin(ti * 12.9898 + l1 * 78.233) * 43758.5453
                hsh = hsh - math.floor(hsh)
                up = min(max(-dy * 1.2 + 0.7, 0.25), 1.0)
                tooth = 1.0 - amp * (0.55 + 0.9 * hsh) * up * q ** 1.15
                edge = lump * tooth
                if d >= edge:
                    continue
                qq = d / edge
                h = z0 + r * 0.35 * math.sqrt(max(1.0 - qq * qq, 0.0))
                if h > hb[y, x]:
                    hb[y, x] = h
                    ib[y, x] = i
    return hb, ib


@njit(cache=True, fastmath=True)
def _shade(hb, ib, S, ss, Lx, Ly, Lz, shift, gapd, wl):
    Hs, Ws = hb.shape
    ndl = np.zeros((Hs, Ws), np.float32)
    sh = np.zeros((Hs, Ws), np.float32)
    gp = np.zeros((Hs, Ws), np.float32)
    rim = np.zeros((Hs, Ws), np.float32)
    sw = np.zeros((Hs, Ws), np.float32)
    ll = math.sqrt(Lx * Lx + Ly * Ly) + 1e-9
    ux, uy = Lx / ll, Ly / ll
    gi = max(int(gapd), 1)
    for y in range(Hs):
        for x in range(Ws):
            i = ib[y, x]
            if i < 0:
                continue
            cx, cy = S[i, 0] * ss, S[i, 1] * ss
            r = S[i, 2] * ss
            ry = r * S[i, 3]
            dx = (x + 0.5 - cx) / r
            dy = (y + 0.5 - cy) / ry
            dd = dx * dx + dy * dy
            if dd > 0.95:
                s = math.sqrt(0.95 / dd)
                dx *= s
                dy *= s
                dd = 0.95
            nz = math.sqrt(1.0 - dd)
            gx, gy = S[i, 5] * ss, S[i, 6] * ss
            grx, gry = S[i, 7] * ss, S[i, 8] * ss
            ex = (cx - gx) / grx
            ey = (cy - gy) / gry
            sw[y, x] = min(max(0.5 - (y + 0.5 - gy) / gry * 0.5, 0.0), 1.0)
            ee = ex * ex + ey * ey
            if ee > 0.96:
                s2 = math.sqrt(0.96 / ee)
                ex *= s2
                ey *= s2
                ee = 0.96
            ez = math.sqrt(1.0 - ee)
            Nx = wl * dx + (1.0 - wl) * ex
            Ny = wl * dy + (1.0 - wl) * ey
            Nz = wl * nz + (1.0 - wl) * ez
            nl = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-6
            ndl[y, x] = (Nx * Lx + Ny * Ly + Nz * Lz) / nl + S[i, 18]
            h0 = hb[y, x]
            # short cast shadow from a nearer mass toward the light
            occ = 0.0
            for st in range(1, 6):
                d = shift * st / 5.0
                xx = int(x + ux * d)
                yy = int(y + uy * d)
                if xx < 0 or yy < 0 or xx >= Ws or yy >= Hs:
                    break
                j = ib[yy, xx]
                if j >= 0 and j != i and hb[yy, xx] > h0 + 0.5:
                    occ = 1.0
                    break
            sh[y, x] = occ
            # dark contact gap right above a nearer mass's toothed top edge
            g = 0.0
            for k in range(1, gi + 1):
                yy = y + k
                if yy >= Hs:
                    break
                j = ib[yy, x]
                if j >= 0 and j != i and hb[yy, x] > h0 + 0.5:
                    g = 1.0 - (k - 1) / (gi + 1.0)
                    break
            gp[y, x] = g
            # sun-side outer rim
            xr = int(x + ux * 1.6 * ss)
            yr = int(y + uy * 1.6 * ss)
            if xr >= 0 and yr >= 0 and xr < Ws and yr < Hs:
                if ib[yr, xr] < 0:
                    rim[y, x] = 1.0
    return ndl, sh, gp, rim, sw


def _row(x, y, r, asp, z, g, tone, fog, rng, tree, nt=None, amp=None, flat=0.5, bias=0.0):
    nt = rng.uniform(8, 13) if nt is None else nt
    amp = rng.uniform(0.2, 0.32) if amp is None else amp
    return [x, y, r, asp, z, g[0], g[1], g[2], g[3], float(np.clip(tone, -1, 1)), fog, nt, rng.uniform(0, 1), amp,
            rng.uniform(0, 6.28), rng.uniform(0, 6.28), flat, tree, bias]


_LDIR = np.array([0.62, -0.7, 0.35]) / np.linalg.norm([0.62, -0.7, 0.35])


def _ell_ndl(ex, ey):
    ee = min(ex * ex + ey * ey, 0.96)
    s = math.sqrt(ee / max(ex * ex + ey * ey, 1e-9))
    ex, ey = ex * s, ey * s
    return ex * _LDIR[0] + ey * _LDIR[1] + math.sqrt(1 - ee) * _LDIR[2]


def crown(rng, cx, by, w, h, z0, fog, tone0=0.0, kind='round', leaf_r=None, tree=0.0):
    """One broadleaf crown (camphor / keyaki / shrub) as many small serrated leaf masses grouped into
    3-7 sub-crown clusters. Each cluster is lit as one small volume (lit top-right, shadow bottom-left), the
    whole crown adds a gentle overall light gradient; nearer clusters overlap the ones behind with a hard
    edge. cx, by - crown base (px); w, h - crown size (px); z0 - depth offset (bigger = nearer); leaf_r - mass
    radius (px, default ~w/10). Returns a list of clump rows."""
    r0 = (w / 10.0 if leaf_r is None else leaf_r)
    r0 = max(r0, 1.6)
    gx, gy = cx, by - h * 0.5
    grx, gry = w * 0.5, h * 0.5

    def half_w(v):
        if kind == 'vase':
            return w * 0.5 * (0.55 + 0.45 * math.sin(math.pi * min(0.15 + 0.75 * v, 1.0))) * (1.0 - 0.35 * v)
        if kind == 'shrub':
            return w * 0.5 * math.sqrt(max(1 - ((v - 0.55) / 0.6) ** 2, 0.0))
        return w * 0.5 * math.sqrt(max(1 - ((v - 0.5) / 0.5) ** 2, 0.0)) ** 0.8

    def inside(x, y):
        v = (y - (by - h)) / h                        # 0 top .. 1 bottom
        if v < 0 or v > 1:
            return -1.0
        return 1.0 - abs(x - cx) / max(half_w(v), 1e-3)

    # sub-crown clusters (sizes relative to the crown), farthest first
    ncl = {'shrub': 3, 'vase': 6}.get(kind, 5) + int(rng.integers(0, 3))
    cl = []
    tries = 0
    while len(cl) < ncl and tries < 400:
        tries += 1
        x = rng.uniform(cx - w * 0.42, cx + w * 0.42)
        y = rng.uniform(by - h * 0.9, by - h * 0.2)
        if inside(x, y) < 0.2:
            continue
        if any(((x - q[0]) / w) ** 2 + ((y - q[1]) / h) ** 2 < 0.035 for q in cl):
            continue
        rx = w * rng.uniform(0.16, 0.34)
        cl.append([x, y, rx, rx * rng.uniform(0.7, 0.95)])
    if not cl:
        cl = [[cx, gy, w * 0.4, h * 0.4]]
    # depth of each cluster: lower and nearer the crown front = nearer
    for q in cl:
        ex, ey = (q[0] - gx) / grx, (q[1] - gy) / gry
        q.append(math.sqrt(max(1 - min(ex * ex + ey * ey, 1.0), 0.0)) * grx * 0.5 + (q[1] - (by - h)) / h * r0 * 2.0
                 + rng.uniform(0, r0))
    n_try = int(5.0 * w * h / (r0 * r0)) + 20
    pts = []
    for _ in range(n_try):
        x = rng.uniform(cx - w * 0.55, cx + w * 0.55)
        y = rng.uniform(by - h * 1.02, by)
        f = inside(x, y)
        if f < -0.05:
            continue
        pts.append((x, y))
    rng.shuffle(pts)
    keep = []
    md = r0 * 0.62
    grid = {}
    for (x, y) in pts:
        gxk, gyk = int(x // md), int(y // md)
        ok = True
        for ii in (-1, 0, 1):
            for jj in (-1, 0, 1):
                for (qx, qy) in grid.get((gxk + ii, gyk + jj), ()):
                    if (qx - x) ** 2 + (qy - y) ** 2 < md * md:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append((x, y))
            grid.setdefault((gxk, gyk), []).append((x, y))
    out = []
    for (x, y) in keep:
        v = (y - (by - h)) / h
        # nearest cluster (normalised distance); masses outside every cluster form the dim inner body
        best, bk = 1e9, -1
        for k, q in enumerate(cl):
            d = ((x - q[0]) / q[2]) ** 2 + ((y - q[1]) / q[3]) ** 2
            if d < best:
                best, bk = d, k
        q = cl[bk]
        # size hierarchy: big masses in the heart of each cluster, small serrated ones breaking up the
        # cluster edges (and the crown silhouette); the dim inner body is medium-sized
        ef = min(best, 1.0)
        r = r0 * math.exp(rng.normal(0.0, 0.25)) * ((1.45 - 0.75 * ef) if best <= 1.0 else 1.0) * (0.9 + 0.2 * v)
        cb = _ell_ndl((x - gx) / grx, (y - gy) / gry)          # whole-crown light at this mass
        if best <= 1.0:
            g = (q[0], q[1], q[2] * 1.1, q[3] * 1.1)
            zc = q[4] + math.sqrt(max(1 - best, 0.0)) * q[2] * 0.5
            bias = 0.22 * cb - 0.14
        else:
            g = (gx, gy, grx * 1.05, gry * 1.05)
            zc = math.sqrt(max(1 - min(((x - gx) / grx) ** 2 + ((y - gy) / gry) ** 2, 1.0), 0)) * grx * 0.2 - r0
            bias = -0.5
        out.append(_row(x, y, r, rng.uniform(0.62, 0.8), z0 + zc + rng.uniform(0, r0 * 0.3), g,
                        tone0 + rng.normal(0, 0.1), fog, rng, tree, bias=bias,
                        amp=rng.uniform(0.26, 0.36) if r < r0 * 0.9 else None))
    return out


def hedge_row(rng, x0, x1, ybase, top_fn, r0, z0, fog, tone_sd=0.2, tree=0.0):
    """A continuous band of small serrated masses (hedgerow / far wooded hill), top edge following top_fn(x)."""
    out = []
    x = x0
    k = 0
    while x < x1:
        top = top_fn(x)
        hgt = max(ybase - top, r0)
        g = (x, top + hgt * 0.5, max(hgt * 0.9, r0 * 3), hgt * 0.55 + r0)
        y = top + r0 * 0.6
        lev = 0
        tone = rng.normal(0, tone_sd)
        while y < ybase + r0 * 0.4:
            rr = r0 * rng.uniform(0.8, 1.2) * (1 + 0.12 * lev)
            out.append(_row(x + rng.normal(0, 0.35) * r0, y, rr, rng.uniform(0.6, 0.78), z0 + lev * r0 * 0.6 +
                            rng.uniform(0, r0 * 0.3), g, tone + rng.normal(0, 0.1), fog, rng, tree + k * 0.01))
            y += rr * rng.uniform(0.7, 1.0)
            lev += 1
        x += r0 * rng.uniform(0.9, 1.3)
        k += 1
    return out


def render(W, H, S, pal=None, light=(0.62, -0.7, 0.35), ss=2, shift=None, wl=0.2, holes=1.0, seed=0,
           rim_amt=0.6):
    """Render clumps -> (rgba (H, W, 4) straight alpha, sway weight (H, W))."""
    pal = PAL if pal is None else pal
    L = np.asarray(light, np.float64)
    L = L / np.linalg.norm(L)
    S = np.ascontiguousarray(np.asarray(S, np.float32))
    Hs, Ws = H * ss, W * ss
    hb, ib = _zbuf(Hs, Ws, S, ss)
    mr = float(np.median(S[:, C_R])) if len(S) else 4.0
    shift = mr * 0.5 * ss if shift is None else shift * ss
    ndl, sh, gp, rim, sw = _shade(hb, ib, S, ss, float(L[0]), float(L[1]), float(L[2]), float(shift),
                                  max(mr * 0.22 * ss, 1.0), wl)
    cov = (ib >= 0).astype(np.float32)
    idx = np.maximum(ib, 0)
    tone = np.where(ib >= 0, S[idx, C_TONE], 0).astype(np.float32)
    fog = np.where(ib >= 0, S[idx, C_FOG], 0).astype(np.float32)
    # leaf-scale dither at the value steps (a broken, painted terminator, not a smooth ramp)
    N = P.tile_noise(256, seed=600 + seed, octaves=3, base=16).astype(np.float32)
    xs = (np.arange(Ws, dtype=np.float32) / (mr * ss * 0.7))[None, :].repeat(Hs, 0)
    ys = (np.arange(Hs, dtype=np.float32) / (mr * ss * 0.7))[:, None].repeat(Ws, 1)
    dn = P.sample_tile(N, xs, ys) - 0.5
    v = ndl + 0.12 * tone + dn * 0.16 - 0.42 * sh
    # 3 values (+ warm/cool variants and a sparse highlight on the lit tops)
    shd = pal['shd'] + (pal['shd_w'] - pal['shd']) * np.clip(tone + 0.3, 0, 1)[..., None]
    mid = pal['mid'] + (pal['mid_w'] - pal['mid']) * np.clip(tone + 0.3, 0, 1)[..., None]
    core = pal.get('core', pal['shd'])
    col = core + (shd - core) * sstep(-0.3, -0.24, v)[..., None]
    col = col + (mid - col) * sstep(0.08, 0.13, v)[..., None]
    col = col + (pal['lit'] - col) * sstep(0.47, 0.52, v)[..., None]
    col = col + (pal['hi'] - col) * (sstep(0.72, 0.76, v) * 0.8)[..., None]
    col = col + (pal['gap'] - col) * (gp * 0.45)[..., None]
    rimv = rim * sstep(0.0, 0.25, ndl) * (1 - gp)
    col = col + (pal['rim'] - col) * (rimv * rim_amt)[..., None]
    col = col + (pal['haze'] - col) * fog[..., None]
    a = cov
    if holes > 0:
        m8 = (cov > 0.5).astype(np.uint8)
        din = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        N2 = P.tile_noise(256, seed=610 + seed, octaves=2, base=40).astype(np.float32)
        hn = P.sample_tile(N2, xs * 5.0 + 11, ys * 5.0 + 5)
        lr = mr * ss
        hole = (din > 0.4 * lr) & (din < 1.8 * lr) & (hn > 1 - 0.07 * holes) & (rim < 0.5)
        a = np.where(hole, 0.0, a).astype(np.float32)
    pm = np.dstack([col * a[..., None], a, sw * a])
    small = cv2.resize(pm.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    rgba = np.dstack([small[..., :3] / np.maximum(al, 1e-4)[..., None], al]).astype(np.float32)
    sway = (small[..., 4] / np.maximum(al, 1e-4)).astype(np.float32)
    return rgba, sway
