"""s05_sakura helpers: canopy clean-up passes on the supersampled tree z-buffers.

* prune_fragments: bark primitives that only show up as small disconnected slivers between blossom masses
  (the 'stray black dashes') are removed and the z-buffer is re-rendered without them.
* entry_florets: where a limb disappears behind a nearer blossom mass, a bunch of individual flowers is
  painted over the entry so the wood visibly grows INTO the blossom instead of ending in a sawn-off cap.
* draw_florets: numba stamp of small five-petal flowers with per-flower colour (premultiplied RGBA).
"""
import math
import numpy as np
import cv2
from numba import njit

import s05_sakura_raster as R


@njit(cache=True, fastmath=True)
def draw_twigs(buf, segs, dark, lit, rimc, lx, ly):
    """tapered wood strokes: segs rows x0 y0 r0 x1 y1 r1.  Two-tone (shadow / lit side toward (lx, ly))
    with a thin warm rim on the sun side."""
    H, W = buf.shape[0], buf.shape[1]
    for i in range(segs.shape[0]):
        ax, ay, ar = segs[i, 0], segs[i, 1], segs[i, 2]
        bx, by, br = segs[i, 3], segs[i, 4], segs[i, 5]
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
                t = min(max((px * ex + py * ey) / L2, 0.0), 1.0)
                qx = px - t * ex
                qy = py - t * ey
                rr = ar + (br - ar) * t
                d = math.sqrt(qx * qx + qy * qy)
                cov = min(max(rr + 0.5 - d, 0.0), 1.0)
                if cov <= 0.0:
                    continue
                sg = min(max((qx * nx + qy * ny) / max(rr, 0.5), -1.0), 1.0)
                facing = (nx * lx + ny * ly) * sg
                k = 0.0
                if facing > 0.25:
                    k = 1.0
                r = dark[0] + (lit[0] - dark[0]) * k * 0.6
                g = dark[1] + (lit[1] - dark[1]) * k * 0.6
                b = dark[2] + (lit[2] - dark[2]) * k * 0.6
                rim = 0.0
                if abs(sg) > 0.6 and facing > 0.3:
                    rim = 0.8
                r += (rimc[0] - r) * rim
                g += (rimc[1] - g) * rim
                b += (rimc[2] - b) * rim
                buf[y, x, 0] = buf[y, x, 0] * (1 - cov) + r * cov
                buf[y, x, 1] = buf[y, x, 1] * (1 - cov) + g * cov
                buf[y, x, 2] = buf[y, x, 2] * (1 - cov) + b * cov
                if buf.shape[2] > 3:
                    buf[y, x, 3] = buf[y, x, 3] * (1 - cov) + cov


@njit(cache=True, fastmath=True)
def draw_florets(buf, fl):
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
                r = cr * (1.0 - 0.02 * base)
                g = cg * (1.0 - 0.14 * base)
                b = cb * (1.0 - 0.06 * base)
                e = cen * (1.0 - min(max((t - 0.08) / 0.1, 0.0), 1.0))
                r += (0.8 - r) * e
                g += (0.32 - g) * e
                b += (0.48 - b) * e
                # thin darker petal outline
                ol = (1.0 - min(max((rp - rho) / 1.2, 0.0), 1.0)) * 0.18
                r *= 1.0 - ol * 0.6
                g *= 1.0 - ol
                b *= 1.0 - ol * 0.5
                buf[y, x, 0] = buf[y, x, 0] * (1 - cov) + r * cov
                buf[y, x, 1] = buf[y, x, 1] * (1 - cov) + g * cov
                buf[y, x, 2] = buf[y, x, 2] * (1 - cov) + b * cov
                if nc > 3:
                    buf[y, x, 3] = buf[y, x, 3] * (1 - cov) + cov


def prune_fragments(ib, P_local, amin, rz_max=0.045):
    """-> array of local primitive indices (bark) that are only visible as small slivers."""
    valid = ib >= 0
    mat = np.where(valid, P_local[np.maximum(ib, 0), R.K_MAT], -1)
    bark = (mat == 2).astype(np.uint8)
    if not bark.any():
        return np.zeros(0, np.int64)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bark, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    small = area < amin
    small[0] = False
    sm = small[lab] & (bark > 0)
    kill = np.unique(ib[sm])
    kill = kill[np.maximum(P_local[kill, R.K_RZ0], P_local[kill, R.K_RZ1]) < rz_max]
    keep = np.unique(ib[(~small[lab]) & (bark > 0)])
    return np.setdiff1d(kill, keep)


def entry_florets(out, zb, ib, P, seed, fpx, light_dir=(-0.9, -0.4), prims=None, off=(0.0, 0.0), bark=None,
                  haze=None, rz_trunk=0.125):
    """Where a limb (walking along its axis) passes from visible wood into a nearer blossom mass, paint the
    limb splitting into 2-3 tapering forks that run on over the blossom, and flower bunches over the entry and
    the fork tips (the wood grows INTO the blossom instead of ending in a sawn-off cap).
    out: (h, w, 4) straight RGBA shaded buffer (ss px), P: primitive table indexed by ib (ss px coords already
    local to this buffer for the rows listed in prims).  fpx: flower radius in ss px."""
    h, w = ib.shape
    valid = ib >= 0
    mat = np.where(valid, P[np.maximum(ib, 0), R.K_MAT], -1)
    blos = (mat == 0) | (mat == 1)
    if prims is None or len(prims) == 0 or not blos.any():
        return
    rng = np.random.default_rng(1300 + seed)
    ents = []
    prims = np.asarray(prims)
    prims = prims[(P[prims, R.K_KIND] == 1) & (P[prims, R.K_MAT] == 2) &
                  (np.maximum(P[prims, R.K_R0], P[prims, R.K_R1]) >= 1.8) &
                  (np.maximum(P[prims, R.K_RZ0], P[prims, R.K_RZ1]) < rz_trunk)]
    for i in prims:
        x0, y0, r0 = P[i, R.K_X0] - off[0], P[i, R.K_Y0] - off[1], P[i, R.K_R0]
        x1, y1, r1 = P[i, R.K_X1] - off[0], P[i, R.K_Y1] - off[1], P[i, R.K_R1]
        if max(r0, r1) < 1.8:
            continue
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1:
            continue
        # orient from the thicker (trunk) end to the thinner end
        if r1 > r0:
            x0, y0, r0, x1, y1, r1 = x1, y1, r1, x0, y0, r0
        ext = (1.5 * r1 + 2.0) / L
        if (y0 - y1) < 0.15 * L:
            continue                  # only limbs that grow upward / outward (never roots)
        n = int(L * (1 + ext)) + 2
        tt = np.linspace(0, 1 + ext, n)
        xf = x0 + (x1 - x0) * tt
        yf = y0 + (y1 - y0) * tt
        inside = (xf >= 0) & (xf < w) & (yf >= 0) & (yf < h)
        xs = np.clip(xf.astype(int), 0, w - 1)
        ys = np.clip(yf.astype(int), 0, h - 1)
        m = np.where(inside, mat[ys, xs], -2)
        wood = (m == 2) & (tt <= 1.0 + 0.5 * ext)
        bl = (m == 0) | (m == 1) | ((m == -1) & (tt > 0.9))      # into blossom, or a visible end in the air
        tr = np.nonzero(wood[:-1] & bl[1:])[0]
        for k in tr:
            rr = r0 + (r1 - r0) * min(tt[k], 1.0)
            if rr < 1.8:
                continue
            ents.append((xs[k], ys[k], (x1 - x0) / L, (y1 - y0) / L, rr))
    import os
    if os.environ.get('DBG_ENT'):
        print('entry prims', len(prims), 'ents', len(ents))
    if not ents:
        return
    # de-duplicate entries closer than their limb width
    ents.sort(key=lambda e: -e[4])
    kept = []
    for e in ents:
        if all(math.hypot(e[0] - q[0], e[1] - q[1]) > 2.2 * max(e[4], q[4]) for q in kept):
            kept.append(e)
    bm = blos.astype(np.float32)
    sig = max(2.0, 2.0 * fpx)
    col = cv2.GaussianBlur(out[..., :3] * bm[..., None], (0, 0), sig) /         np.maximum(cv2.GaussianBlur(bm, (0, 0), sig), 1e-3)[..., None]
    fl = []
    segs = []
    for (x0, y0, dx, dy, rr) in kept:
        c = col[int(y0), int(x0)]
        lum = float(c.mean())
        centres = [(x0 + dx * rr * 0.9, y0 + dy * rr * 0.9, rr * 1.9)]
        nfk = int(rng.integers(2, 4))
        for m in range(nfk):
            ang = (m - (nfk - 1) / 2) * rng.uniform(0.45, 0.7) + rng.normal(0, 0.12)
            ca, sa = math.cos(ang), math.sin(ang)
            fx, fy = dx * ca - dy * sa, dx * sa + dy * ca
            L = rr * rng.uniform(4.0, 7.0)
            f0 = rr * rng.uniform(0.45, 0.62)
            bend = rng.normal(0, 0.3)
            sx, sy = x0 - dx * rr * 0.6, y0 - dy * rr * 0.6
            mx_ = x0 + fx * L * 0.5 - fy * L * 0.5 * bend * 0.3
            my_ = y0 + fy * L * 0.5 + fx * L * 0.5 * bend * 0.3
            ex_ = x0 + fx * L - fy * L * bend * 0.45
            ey_ = y0 + fy * L + fx * L * bend * 0.45
            f1 = max(0.7, f0 * 0.42)
            segs.append((sx, sy, f0, mx_, my_, f1, 0.0, L * 0.5))
            segs.append((mx_, my_, f1, ex_, ey_, 0.45, 0.0, L * 0.5))
            # a tiny side twig off the fork
            if rng.random() < 0.6:
                ta = ang + rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
                tx_, ty_ = dx * math.cos(ta) - dy * math.sin(ta), dx * math.sin(ta) + dy * math.cos(ta)
                segs.append((mx_, my_, f1 * 0.8, mx_ + tx_ * L * 0.4, my_ + ty_ * L * 0.4, 0.4, 0.0, L * 0.4))
                centres.append((mx_ + tx_ * L * 0.4, my_ + ty_ * L * 0.4, max(rr * 0.5, 1.8 * fpx)))
            centres.append((ex_, ey_, max(rr * 0.6, 2.4 * fpx)))
            centres.append((mx_ + fx * rr * 0.3 + fy * rr * 0.4, my_ + fy * rr * 0.3 - fx * rr * 0.4, max(rr * 0.3, 1.4 * fpx)))
        for (cx0, cy0, crad) in centres:
            n = int(rng.integers(4, 8) + min(110, 0.9 * (crad / fpx) ** 2))
            yy = int(np.clip(cy0, 0, h - 1))
            xx = int(np.clip(cx0, 0, w - 1))
            cc0 = col[yy, xx]
            for m in range(n):
                a = rng.uniform(0, 2 * math.pi)
                d = crad * math.sqrt(rng.random())
                x = cx0 + math.cos(a) * d
                y = cy0 + math.sin(a) * d * 0.85
                up = max(0.0, -(math.sin(a)))          # upper florets catch more light
                lift = (0.0 + 0.07 * up) * rng.uniform(0.6, 1.4)
                cc = np.clip(cc0 * (0.97 + lift), 0, 1.08)
                fl.append((x, y, fpx * rng.uniform(0.8, 1.35), rng.uniform(0, 6.283), rng.uniform(0.6, 1.0),
                           rng.uniform(0, 3.14), cc[0], cc[1], cc[2], 0.06))
    segs = []      # (the painted fork twigs read as dotted lines at this scale: flowers only)
    # a blossom bunch hanging over every entry: the limb disappears behind flowers, never in a straight cut
    from s05_sakura_paint import _stamp, _offsets
    offs, oidx = _offsets(rng, 24)
    dabs = []
    for (x0, y0, dx, dy, rr) in kept:
        for j in range(int(rng.integers(2, 4))):
            cx0 = x0 + dx * rr * rng.uniform(0.2, 1.4) + rng.normal(0, rr * 0.5)
            cy0 = y0 + dy * rr * rng.uniform(0.2, 1.4) + rng.normal(0, rr * 0.5)
            c = col[int(np.clip(cy0, 0, h - 1)), int(np.clip(cx0, 0, w - 1))]
            lk = float(np.clip((c.mean() - 0.45) / 0.4, 0, 1))
            cc = np.clip(c * (0.98 + 0.04 * j), 0, 1.08)
            s_ = int(rng.integers(len(oidx)))
            dabs.append((cx0, cy0, rr * rng.uniform(1.3, 2.2), oidx[s_][1], oidx[s_][0], cc[0], cc[1], cc[2],
                         max(lk, 0.3), 0.5, 0))
    if dabs:
        dabs = np.array(dabs, np.float64)
        dabs = dabs[np.argsort(-dabs[:, 2])]
        buf3 = np.ascontiguousarray(out[..., :3])
        am = np.ones((h, w), np.float32)
        before = buf3.copy()
        _stamp(buf3, am, dabs, offs, float(light_dir[0]), float(light_dir[1]))
        chg = (np.abs(buf3 - before).sum(-1) > 0).astype(np.float32)
        out[..., :3] = buf3
        out[..., 3] = np.maximum(out[..., 3], chg)
    if segs:
        dk = np.array(bark[0] if bark is not None else (0.08, 0.06, 0.09), np.float64)
        lt = np.array(bark[1] if bark is not None else (0.34, 0.22, 0.18), np.float64)
        rc = np.array((1.0, 0.72, 0.48), np.float64)
        if haze is not None:
            hc, hk = haze
            dk = dk + (np.asarray(hc) - dk) * hk
            lt = lt + (np.asarray(hc) - lt) * hk
            rc = rc + (np.asarray(hc) - rc) * hk
        sg = np.array(segs, np.float64)[:, :6]
        draw_twigs(out, sg, dk, lt, rc, float(light_dir[0]), float(light_dir[1]))
    if fl:
        fl = np.array(fl, np.float64)
        fl = fl[np.argsort(fl[:, 1])]
        draw_florets(out, fl)


def shift_rep(a, dy, dx):
    """shift a 2D array by (dy, dx) px with edge replication (no fake border)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
