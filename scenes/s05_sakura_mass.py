"""s05_sakura helpers: 'big painted blossom masses' repaint of the tree cards (round 2).

The 3D dome z-buffer (trees + raster) gives the canopy its silhouettes and depth, but shading every
dome/floret produced popcorn / bubble-wrap and noisy magenta mush.  This pass repaints every blossom pixel
the way a background artist paints a cherry crown:

* the clumps of each tree are grouped (k-means in camera space) into a FEW big masses (2-5 per tree);
* each mass is lit as ONE form: its own ellipsoid normal blended with the tree's crown ellipsoid, dotted
  with the light -> pink-white lit tops, cool lavender undersides.  Values are posterised into 5 flat
  tones with firm, slightly scalloped terminators (the floret relief only perturbs the threshold, so lit
  flower heads bulge into the shadow but never speckle the interiors);
* nearer masses cast crisp shadows on the masses behind them (overlap edges read as painted value steps);
* thin, sun-facing shadow passages glow with warm transmitted light (translucent petals);
* aerial haze by depth.
`lace()` then breaks the silhouette with 5-petal flower clusters, pinholes of sky and lit edge flowers.
"""
import math
import numpy as np
import cv2

from s05_sakura_raster import K_KIND, K_X0, K_Y0, K_Z0, K_R0, K_PAR, K_TREE, K_MAT, K_X1, K_Y1, K_R1
from numba import njit

from s05_sakura_canopy import shift_rep


@njit(cache=True, fastmath=True)
def florets(buf, fl, ol_amt=0.08, base_amt=0.6):
    """fl rows: x, y, R, rot, squash, tilt, r, g, b, centre_amt.  buf (h, w, 3|4) straight colour written
    as 'over' with coverage (alpha channel accumulated if present)."""
    H, W = buf.shape[0], buf.shape[1]
    nc = buf.shape[2]
    for i in range(fl.shape[0]):
        cx, cy, Rr = fl[i, 0], fl[i, 1], fl[i, 2]
        rot, sq, tl = fl[i, 3], fl[i, 4], fl[i, 5]
        cr, cg, cb, cen = fl[i, 6], fl[i, 7], fl[i, 8], fl[i, 9]
        x0 = max(0, int(cx - Rr - 2))
        x1 = min(W, int(cx + Rr + 3))
        y0 = max(0, int(cy - Rr - 2))
        y1 = min(H, int(cy + Rr + 3))
        ct = math.cos(tl)
        st = math.sin(tl)
        for y in range(y0, y1):
            for x in range(x0, x1):
                dx = x + 0.5 - cx
                dy = y + 0.5 - cy
                u = dx * ct + dy * st
                v = (-dx * st + dy * ct) / sq
                rho = math.sqrt(u * u + v * v)
                th = math.atan2(v, u) - rot
                sec = 2.0 * math.pi / 5.0
                k = math.floor(th / sec + 0.5)
                ph = th - k * sec
                cph = math.cos(ph * 2.2)
                if cph <= 0.0:
                    rp = 0.35 * Rr
                else:
                    rp = Rr * (0.35 + 0.65 * cph ** 0.5)
                rp *= 1.0 - 0.14 * math.exp(-(ph / 0.08) ** 2)
                cov = min(max(rp - rho + 0.5, 0.0), 1.0)
                if cov <= 0.0:
                    continue
                t = rho / Rr
                # pale petal tips, pinker base, tiny dark-pink eye
                base = 1.0 - (min(max((t - 0.1) / 0.45, 0.0), 1.0))
                r = cr * (1.0 - 0.02 * base * base_amt)
                g = cg * (1.0 - 0.14 * base * base_amt)
                b = cb * (1.0 - 0.06 * base * base_amt)
                e = cen * (1.0 - min(max((t - 0.08) / 0.1, 0.0), 1.0))
                r += (0.8 - r) * e
                g += (0.32 - g) * e
                b += (0.48 - b) * e
                # thin darker petal outline
                ol = (1.0 - min(max((rp - rho) / 1.2, 0.0), 1.0)) * ol_amt
                r *= 1.0 - ol * 0.6
                g *= 1.0 - ol
                b *= 1.0 - ol * 0.5
                buf[y, x, 0] = buf[y, x, 0] * (1 - cov) + r * cov
                buf[y, x, 1] = buf[y, x, 1] * (1 - cov) + g * cov
                buf[y, x, 2] = buf[y, x, 2] * (1 - cov) + b * cov
                if nc > 3:
                    buf[y, x, 3] = buf[y, x, 3] * (1 - cov) + cov



# ============================================================================ mass assignment

def assign_masses(P, f_ss, per_mass=7, kmin=2, kmax=5, seed=0):
    """-> int array (n,) : mass id per primitive (-1 for bark).  Mass ids are unique across trees."""
    n = len(P)
    sph = P[:, K_KIND] == 0
    root = np.arange(n)
    for _ in range(4):
        par = np.where(sph[root], P[root, K_PAR], -1).astype(np.int64)
        root = np.where(par >= 0, par, root)
    mass = np.full(n, -1, np.int64)
    Z = np.maximum(P[:, K_Z0], 1e-3)
    X = P[:, K_X0] * Z / f_ss
    Y = P[:, K_Y0] * Z / f_ss
    rw = P[:, K_R0] * Z / f_ss
    is_root = sph & (P[:, K_PAR] < 0) & (P[:, K_R0] > 0)
    rng = np.random.default_rng(seed)
    nxt = 0
    for t in np.unique(P[is_root, K_TREE]).astype(int):
        ri = np.nonzero(is_root & (P[:, K_TREE] == t))[0]
        # weight by clump size: tiny tufts must not seed their own mass
        w = rw[ri] ** 2 + 1e-6
        big = ri[rw[ri] > np.percentile(rw[ri], 40)] if len(ri) > 4 else ri
        k = int(np.clip(round(len(big) / per_mass), kmin, kmax))
        k = min(k, len(ri))
        pts = np.stack([X[ri], Y[ri] * 1.15, Z[ri] * 0.45], 1)
        # deterministic farthest-point init from the biggest clump
        c0 = [int(np.argmax(w))]
        d = np.linalg.norm(pts - pts[c0[0]], axis=1)
        for _ in range(1, k):
            j = int(np.argmax(d * np.sqrt(w)))
            c0.append(j)
            d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
        cen = pts[c0].copy()
        for _ in range(12):
            lab = np.argmin(((pts[:, None, :] - cen[None]) ** 2).sum(-1), 1)
            for j in range(k):
                s = lab == j
                if s.any():
                    cen[j] = (pts[s] * w[s, None]).sum(0) / w[s].sum()
        mass[ri] = nxt + lab
        nxt += k
    mass[sph] = mass[root[sph]]
    return mass


# ============================================================================ helpers

def _nblur(v, m, sig):
    if sig < 0.3:
        return v
    num = cv2.GaussianBlur(v * m, (0, 0), sig)
    den = cv2.GaussianBlur(m, (0, 0), sig)
    return num / np.maximum(den, 1e-4)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(h, w, cell, seed, oct_=2):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(oct_):
        c = max(2.0, cell / (2 ** o))
        gh, gw = int(h / c) + 3, int(w / c) + 3
        g = rng.random((gh, gw)).astype(np.float32)
        acc += amp * cv2.resize(g, (int(gw * c), int(gh * c)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        tot += amp
        amp *= 0.5
    return acc / tot


_TEX = {}


def flower_tex(h, w, R, seed=0, dens=1.6):
    """Static flower-texture field (h, w) around 0: overlapping 5-petal flowers of radius ~R with random
    light/dark values; used to modulate the flat value zones so they read as masses of blossoms."""
    R = round(R * 2) / 2
    key = (R, seed % 4)
    if key in _TEX:
        t = _TEX[key]
        reps = (h // t.shape[0] + 1, w // t.shape[1] + 1)
        return np.tile(t, reps)[:h, :w]
    rng = np.random.default_rng(seed % 4)
    T = 192
    tile = np.zeros((T, T, 3), np.float32)
    n = int(dens * (T / R) ** 2)
    xs = rng.uniform(-R, T + R, n)
    ys = rng.uniform(-R, T + R, n)
    val = rng.uniform(0.0, 1.0, n)
    fl = np.stack([xs, ys, R * rng.uniform(0.6, 1.1, n), rng.uniform(0, 6.283, n), rng.uniform(0.55, 1.0, n),
                   rng.uniform(0, 3.14, n), val, val, val, np.zeros(n)], 1)
    fl = np.concatenate([fl] + [fl + np.array([dx, dy] + [0] * 8) for dx, dy in
                                ((T, 0), (-T, 0), (0, T), (0, -T), (T, T), (-T, -T), (T, -T), (-T, T))], 0)
    tile[:] = 0.5
    florets(tile, fl.astype(np.float64), 0.25, 0.0)
    t = tile[..., 0] - 0.5
    _TEX[key] = t
    reps = (h // T + 1, w // T + 1)
    return np.tile(t, reps)[:h, :w].astype(np.float32)


# ============================================================================ repaint

def repaint(out, zb, ib, P, mass, pal, L, f_ss, off, haze, haze_k, haze_max, hy_ss, glow=(1.0, 0.68, 0.7),
            seed=0, cast=(0.3, 0.75), detail=0.18, under=0.45, sky=(0.72, 0.78, 1.0), zc=10.0, trans=1.0,
            tree_w=0.35, tex=1.0):
    """Repaint blossom pixels of a supersampled tree card in place.
    out (h, w, 4) straight RGBA from raster.shade; zb, ib (h, w) with GLOBAL primitive indices in ib.
    pal (5, 3): deep, shade, mid, lit, hot.  L: light dir (x right, y up, z toward viewer)."""
    h, w = zb.shape
    offx, offy = off
    valid = ib >= 0
    ibc = np.maximum(ib, 0)
    matp = np.where(valid, P[ibc, K_MAT], -1)
    blos = valid & (matp != 2)
    if not blos.any():
        return
    mp = np.where(blos, mass[ibc], -1)
    bm = blos.astype(np.float32)
    am = valid.astype(np.float32)
    ys = (np.arange(h, dtype=np.float32) + 0.5 + offy)[:, None]
    xs = (np.arange(w, dtype=np.float32) + 0.5 + offx)[None, :]
    Lx, Ly, Lz = L
    # ---- per-mass and per-tree ellipses from the primitives of this card
    ids = np.unique(mp[mp >= 0])
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)
    nz = np.zeros((h, w), np.float32)
    trees = {}
    ell = {}
    order = np.argsort(mass, kind='stable')
    ms = mass[order]
    rows = {}
    for g in ids:
        a0, a1 = np.searchsorted(ms, g, 'left'), np.searchsorted(ms, g, 'right')
        rows[g] = order[a0:a1]
    for g in ids:
        s = rows[g][P[rows[g], K_R0] > 0]
        x, y, r = P[s, K_X0], P[s, K_Y0], P[s, K_R0]
        x0, x1 = (x - r).min(), (x + r).max()
        y0, y1 = (y - r).min(), (y + r).max()
        ell[g] = ((x0 + x1) / 2, (y0 + y1) / 2 + 0.08 * (y1 - y0), (x1 - x0) / 2 * 1.05, (y1 - y0) / 2 * 1.1)
        t = int(P[s, K_TREE][0])
        bb = trees.get(t, [1e9, 1e9, -1e9, -1e9])
        trees[t] = [min(bb[0], x0), min(bb[1], y0), max(bb[2], x1), max(bb[3], y1)]
    tp = np.where(blos, P[ibc, K_TREE], -1).astype(np.int64)
    for g in ids:
        sel = mp == g
        if not sel.any():
            continue
        cx, cy, rx, ry = ell[g]
        yy, xx = np.nonzero(sel)
        ex = (xs[0, xx] - cx) / rx
        ey = -(ys[yy, 0] - cy) / ry
        t = int(P[rows[g][0], K_TREE])
        b = trees[t]
        tcx, tcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 + 0.1 * (b[3] - b[1])
        trx, try_ = (b[2] - b[0]) / 2 * 1.05, (b[3] - b[1]) / 2 * 1.1
        fx = (xs[0, xx] - tcx) / trx
        fy = -(ys[yy, 0] - tcy) / try_
        for a_, b_ in ((ex, ey), (fx, fy)):
            d = a_ * a_ + b_ * b_
            k = np.where(d > 0.92, np.sqrt(0.92 / np.maximum(d, 1e-6)), 1.0)
            a_ *= k
            b_ *= k
        ez = np.sqrt(np.maximum(1 - ex * ex - ey * ey, 0.0))
        fz = np.sqrt(np.maximum(1 - fx * fx - fy * fy, 0.0))
        wt = tree_w
        nx[yy, xx] = (1 - wt) * ex + wt * fx
        ny[yy, xx] = (1 - wt) * ey + wt * fy
        nz[yy, xx] = (1 - wt) * ez + wt * fz
    # masses above the eye are seen from below
    el = np.clip((hy_ss - ys) / f_ss * 2.0, 0, 1)
    ny = ny - under * el * bm
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    lam = (nx * Lx + ny * Ly + nz * Lz) / nn
    # ---- floret relief (from the dome shading) only perturbs the terminator
    lum = out[..., :3].mean(-1)
    Zs = np.where(blos, zb, zc).astype(np.float32)
    fpx = float(np.clip(f_ss * 0.1 / zc, 1.0, 40.0))
    hp = lum - _nblur(lum, bm, fpx * 2.0)
    hp = np.clip(hp, -0.25, 0.25)
    # ---- cast shadows of nearer masses onto the masses behind (screen-space, toward the light)
    zsm = _nblur(Zs, bm, max(1.0, fpx * 0.6))
    zsm = np.where(blos, zsm, 1e4).astype(np.float32)
    ldx, ldy = -Lx, Ly           # screen step toward the light: x right, y down -> (Lx, -Ly)
    ln = math.hypot(Lx, Ly) + 1e-6
    sh_amt = np.zeros((h, w), np.float32)
    for k_, dm in enumerate(cast[:1]):
        dpx = f_ss * dm / zc
        dx, dy = Lx / ln * dpx, -Ly / ln * dpx
        zq = shift_rep(zsm, -dy, -dx)       # value at p + (dx, dy)
        M_ = np.float32([[1, 0, -dx], [0, 1, -dy]])
        mq = cv2.warpAffine(mp.astype(np.float32), M_, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REPLICATE)
        occl = (zq < zsm - 0.15 * dm * max(zc / 10.0, 1.0)) & (mq != mp) & (zq < 1e3)
        sh_amt += occl.astype(np.float32)
    sh_amt = cv2.GaussianBlur(sh_amt, (0, 0), 0.8) * bm
    # ---- sky wrap: silhouette pixels whose sunward side is sky pick up light
    dpx = f_ss * 0.35 / zc
    aq = shift_rep(am, dpx * Ly / ln, -dpx * Lx / ln)
    wrap = (1 - aq) * bm
    wrap = cv2.GaussianBlur(wrap, (0, 0), max(0.6, fpx * 0.4))
    # ---- value
    nzs = _noise(h, w, fpx * 6, seed + 11)
    v = 0.45 + 0.95 * lam + detail * hp * 2.2 + 0.12 * wrap + (nzs - 0.5) * 0.12
    # three big value zones (shade / mid / lit) with firm steps, a soft gradient inside each zone, a small
    # hot zone on the very tops; the cast shadows of nearer masses are a separate crisp deep-lavender pass
    th = np.array([0.38, 0.66, 0.98], np.float32)
    sw = 0.02
    k = np.zeros((h, w), np.float32)
    for tt in th:
        k += _ss(tt - sw, tt + sw, v)
    # the hot zone only where the FORM faces the light (the floret relief never makes white popcorn)
    k -= _ss(th[2] - sw, th[2] + sw, v) * (1 - _ss(th[2] - 0.1, th[2] - 0.0, 0.45 + 0.95 * lam))
    k += 1.0                                   # zone index 1..4 (0 = deep reserved for the cast shadow)
    ki = np.clip(np.floor(k), 0, 4).astype(np.int64)
    kf = (k - np.floor(k)).astype(np.float32)
    colp = pal[np.minimum(ki, 4)] * (1 - kf[..., None]) + pal[np.minimum(ki + 1, 4)] * kf[..., None]
    # in-zone gradient: toward the darker tone at the zone's low end, lighter at its high end
    edges_ = np.concatenate([[-0.3], th, [1.5]]).astype(np.float32)
    zi = np.clip(ki - 1, 0, 3)
    lo_, hi_ = edges_[zi], edges_[zi + 1]
    u = np.clip((v - lo_) / np.maximum(hi_ - lo_, 1e-3), 0, 1) - 0.5
    grad = np.where(u[..., None] < 0, pal[np.maximum(ki - 1, 0)], pal[np.minimum(ki + 1, 4)])
    col = colp + (grad - colp) * (np.abs(u) * 0.35)[..., None]
    # crisp cast shadow of nearer masses: cool deep lavender with a hint of the underlying tone
    csh = np.clip(sh_amt, 0, 1)[..., None]
    col = col + ((pal[0] * 0.75 + col * 0.25) - col) * csh * 0.8
    ki = np.where(sh_amt > 0.5, 0, ki)
    # blossom texture inside the value zones (low contrast: flowers, never bubbles)
    if tex > 0:
        ftx = flower_tex(h, w, float(np.clip(f_ss * 0.04 / zc, 1.5, 16.0)), seed + 3)
        amp = np.where(ki >= 4, 0.04, np.where(ki == 3, 0.06, 0.075)).astype(np.float32) * tex
        col = col * (1 + (ftx * amp)[..., None])
    # sky fill on the up-facing shade, warm transmitted light in thin sunward shadow passages
    shd = 1 - _ss(1.5, 2.5, k)
    shd = np.maximum(shd, np.clip(sh_amt, 0, 1))
    upk = np.clip(ny / nn, 0, 1) * shd * 0.25
    col += (np.array(sky, np.float32) - col) * upk[..., None]
    dsky = cv2.distanceTransform((am > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    thin = np.exp(-dsky / max(1.0, f_ss * 0.45 / zc))
    face = np.clip(0.3 + 0.7 * (nx * Lx - ny * 0.0 + np.maximum(-ny, 0) * 0.5) / nn, 0, 1)
    tr_ = thin * face * shd * trans * 0.55 + 0.18 * shd * trans * (1 - _ss(0.0, 1.0, sh_amt))
    col += (np.array(glow, np.float32) - col) * np.clip(tr_, 0, 0.7)[..., None]
    # aerial perspective
    hk = haze_max * (1 - np.exp(-Zs / haze_k))
    col += (np.array(haze, np.float32) - col) * hk[..., None]
    out[..., :3] = np.where(blos[..., None], col, out[..., :3])
    return ki


# ============================================================================ lacy edges

def lace(out, ib, P, mass, f_ss, zc, pal, seed=0, holes=1.0, edge_fl=1.0, fsz=0.045, hazek=0.0,
         haze=(0.84, 0.9, 1.0), light=(-0.7, -0.7), ki=None, twig=None):
    """Break the blossom silhouettes (supersampled straight RGBA card -> new array): ragged sky pinholes
    near the edges (irregular clusters of small voids), clusters of 5-petal flowers pushed out of every
    silhouette edge (outer edge and hole rims), and bunches of individual flowers riding the lit side of
    the terminators (lit heads bulging into the shade) and the lit tops."""
    h, w = out.shape[:2]
    rng = np.random.default_rng(seed)
    valid = ib >= 0
    ibc = np.maximum(ib, 0)
    blos = valid & (np.where(valid, P[ibc, K_MAT], -1) != 2)
    if not blos.any():
        return out
    a = out[..., 3].copy()
    R = float(np.clip(f_ss * fsz / zc, 1.6, 22.0))          # flower radius (ss px)
    dist0 = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    wood = valid & ~blos
    dwood = cv2.distanceTransform((~wood).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    col = out[..., :3].copy()
    grpf = _noise(h, w, R * 10, seed + 5, 1)
    # ---- sky pinholes: ragged clusters of small voids 2..7 flower radii inside the silhouette
    if holes > 0:
        band_ = blos & (dist0 > 2.0 * R) & (dist0 < 9 * R) & (dwood > 3 * R) & (grpf > 0.5)
        iy, ix = np.nonzero(band_)
        if len(iy):
            n = int(min(len(iy) / (10 * R) ** 2 * holes, 3000))
            pick = rng.choice(len(iy), size=min(n, len(iy)), replace=False)
            m = np.zeros((h, w), np.uint8)
            for kk in pick:
                cx, cy = float(ix[kk]), float(iy[kk])
                nv = int(rng.integers(3, 9))
                for j in range(nv):
                    rr = R * rng.uniform(0.7, 1.6)
                    cx += rng.normal(0, R * 1.1)
                    cy += rng.normal(0, R * 0.8)
                    cv2.circle(m, (int(cx), int(cy)), max(1, int(round(rr))), 255, -1, cv2.LINE_AA)
            hole = m.astype(np.float32) / 255.0
            a = a * (1 - hole)
            # a few dark twigs cross the larger gaps (branches seen through the canopy)
            if twig is not None:
                tm = np.zeros((h, w), np.uint8)
                for kk in pick[: max(1, len(pick) // 3)]:
                    cx, cy = float(ix[kk]), float(iy[kk])
                    ang = rng.uniform(-0.9, 0.9) + (math.pi if rng.random() < 0.5 else 0.0)
                    L_ = R * rng.uniform(3, 7)
                    pts = []
                    x_, y_ = cx - math.cos(ang) * L_ / 2, cy - math.sin(ang) * L_ / 2
                    for j in range(5):
                        pts.append((x_, y_))
                        ang += rng.normal(0, 0.35)
                        x_ += math.cos(ang) * L_ / 4
                        y_ += math.sin(ang) * L_ / 4
                    pts = np.array(pts, np.int32)
                    cv2.polylines(tm, [pts], False, 255, max(1, int(R * rng.uniform(0.12, 0.22))), cv2.LINE_AA)
                tmf = tm.astype(np.float32) / 255.0 * hole
                tmf = np.minimum(tmf, 1.0)
                col = col * (1 - tmf[..., None]) + np.array(twig, np.float32) * tmf[..., None]
                a = np.maximum(a, tmf)
    vis = a > 0.5
    dist = cv2.distanceTransform(vis.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    lum = col.mean(-1)
    fl = []

    def flower(x, y, c, rs=(0.65, 1.0)):
        fl.append((x, y, R * rng.uniform(*rs), rng.uniform(0, 6.283), rng.uniform(0.6, 1.0),
                   rng.uniform(0, 3.14), c[0], c[1], c[2], 0.0))

    # ---- edge flower clusters on every blossom silhouette edge (outer and hole rims), half outside
    sil = blos & vis & (dist < 1.5)
    iy, ix = np.nonzero(sil)
    if len(iy):
        cell = R * 1.3
        key = (iy // cell).astype(np.int64) * 1000003 + (ix // cell).astype(np.int64)
        perm = rng.permutation(len(iy))
        _, first = np.unique(key[perm], return_index=True)
        sel = perm[first]
        g = grpf[iy[sel], ix[sel]]
        sel = sel[(g > 0.3) & (rng.random(len(sel)) < 0.75 * edge_fl)]
        dsm = cv2.GaussianBlur(dist, (0, 0), max(1.0, R * 0.8))
        gx = cv2.Sobel(dsm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dsm, cv2.CV_32F, 0, 1, ksize=3)
        for kk in sel:
            y, x = iy[kk], ix[kk]
            ox, oy = -gx[y, x], -gy[y, x]
            on = math.hypot(ox, oy) + 1e-6
            ox, oy = ox / on, oy / on
            c = col[y, x]
            inner = dist0[y, x] > 1.5          # rim of a sky pinhole: flowers stay inside, holes stay open
            for j in range(int(rng.integers(2, 6)) if not inner else int(rng.integers(0, 3))):
                push = R * (rng.uniform(-0.3, 1.1) if not inner else rng.uniform(-1.2, -0.2))
                cc = c * rng.uniform(0.98, 1.05)
                flower(x + ox * push + rng.normal(0, R * 0.6), y + oy * push + rng.normal(0, R * 0.6), cc)
    # ---- flower bunches on the lit side of the value steps (lit heads bulging into the next tone down)
    #      and a sparse sprinkle on the lit tops
    if ki is not None:
        kif = ki.astype(np.float32)
        lower = cv2.erode(kif, np.ones((3, 3), np.uint8), iterations=max(1, int(R * 0.6)))
        step = blos & vis & (lower < kif) & (dist > R * 1.5) & (ki >= 2)
        iy, ix = np.nonzero(step)
        if len(iy):
            cell = R * 2.2
            key = (iy // cell).astype(np.int64) * 1000003 + (ix // cell).astype(np.int64)
            perm = rng.permutation(len(iy))
            _, first = np.unique(key[perm], return_index=True)
            sel = perm[first]
            sel = sel[(grpf[iy[sel], ix[sel]] > 0.45) & (rng.random(len(sel)) < 0.45)]
            for kk in sel:
                y, x = iy[kk], ix[kk]
                c = col[y, x]
                for j in range(int(rng.integers(4, 9))):
                    flower(x + rng.normal(0, R * 0.55), y + rng.normal(0, R * 0.45), c, (0.75, 1.1))
        cellf = R * 3.0
        gh, gw = int(h / cellf) + 1, int(w / cellf) + 1
        jy = ((np.arange(gh)[:, None] + rng.random((gh, gw))) * cellf).astype(int).ravel()
        jx = ((np.arange(gw)[None, :] + rng.random((gh, gw))) * cellf).astype(int).ravel()
        ok = (jy < h) & (jx < w)
        jy, jx = jy[ok], jx[ok]
        ok = blos[jy, jx] & vis[jy, jx] & (dist[jy, jx] > R) & (grpf[jy, jx] > 0.55) & (ki[jy, jx] >= 3)
        jy, jx = jy[ok], jx[ok]
        keep = rng.random(len(jy)) < 0.35
        for y, x in zip(jy[keep], jx[keep]):
            up = pal[min(int(ki[y, x]) + 1, 4)]
            c = col[y, x] * 0.6 + up * 0.4
            c = c + (np.array(haze, np.float32) - c) * hazek
            for j in range(int(rng.integers(1, 4))):
                flower(x + rng.normal(0, R * 0.6), y + rng.normal(0, R * 0.6), c, (0.6, 0.9))
    buf = np.dstack([col, a]).astype(np.float32).copy()
    if fl:
        fl = np.array(fl, np.float64)
        fl = fl[np.argsort(fl[:, 1])]
        florets(buf, fl)
    return buf
