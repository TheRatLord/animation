"""s05_sakura helpers: painterly blossom-cluster pass for the tree cards.

The 3D dome z-buffer gives the canopy its masses, light and silhouettes, but shaded domes read as 'cotton
balls'.  This pass re-paints the INSIDE of a blossom mass the way a background artist would: with many
flower-cluster dabs of strongly varied size (a few big bunches carrying medium and small ones), each dab a
scalloped union of small round flower heads, flat-coloured from the underlying mass value, with a crisp pale
crescent on its sun side and a thin darker seam on its shadow side.  Dabs stay inside the existing silhouette
(the silhouette itself is broken by the edge flowers / sky holes of decorate_canopy).
"""
import math
import numpy as np
import cv2
from numba import njit

from s05_sakura_raster import sstep, hash2


@njit(cache=True, fastmath=True)
def _stamp(buf, amask, cl, offs, lx, ly):
    """buf (h, w, 3) straight colour, amask (h, w) canopy coverage.  cl rows:
    x, y, Rc, n_sub, off_index, base r g b, lit_k, shade_k, seed.  offs rows: dx, dy, r (unit cluster)."""
    H, W = buf.shape[0], buf.shape[1]
    for i in range(cl.shape[0]):
        cx, cy, Rc = cl[i, 0], cl[i, 1], cl[i, 2]
        ns = int(cl[i, 3])
        o0 = int(cl[i, 4])
        br, bg, bb = cl[i, 5], cl[i, 6], cl[i, 7]
        litk = cl[i, 8]
        shk = cl[i, 9]
        ext = Rc * 1.6 + 2
        x0 = max(0, int(cx - ext))
        x1 = min(W, int(cx + ext + 1))
        y0 = max(0, int(cy - ext))
        y1 = min(H, int(cy + ext + 1))
        for y in range(y0, y1):
            for x in range(x0, x1):
                am = amask[y, x]
                if am < 0.5:
                    continue
                px = x + 0.5 - cx
                py = y + 0.5 - cy
                # nearest-surface sub-head: max of (1 - d/r) over the heads (union of discs)
                best = -1e9
                bnx = 0.0
                bny = 0.0
                for k in range(ns):
                    ox = offs[o0 + k, 0] * Rc
                    oy = offs[o0 + k, 1] * Rc
                    rr = offs[o0 + k, 2] * Rc
                    dx = px - ox
                    dy = py - oy
                    d = math.sqrt(dx * dx + dy * dy)
                    e = rr - d                       # px inside this head
                    if e > best:
                        best = e
                        bnx = dx / (rr + 1e-3)
                        bny = dy / (rr + 1e-3)
                cov = min(max(best + 0.5, 0.0), 1.0) * min((am - 0.5) * 2.0, 1.0)
                if cov <= 0.0:
                    continue
                # head-level light: crescent on the sun side, seam on the far side
                fac = bnx * lx + bny * ly                     # >0 toward the sun
                rad = math.sqrt(bnx * bnx + bny * bny)
                cres = sstep(0.55, 0.75, rad) * sstep(0.15, 0.5, fac)
                seam = sstep(0.72, 0.95, rad) * sstep(0.0, 0.5, -fac)
                # cluster-level light: the whole bunch is lighter on its sun side
                cf = (px * lx + py * ly) / (Rc + 1e-3)
                v = 1.0 + 0.07 * cf * (0.4 + litk)
                r = br * v
                g = bg * v
                b = bb * v
                cr = 0.6 * litk * litk
                r += (1.08 - r) * cres * cr
                g += (0.98 - g) * cres * cr
                b += (0.96 - b) * cres * cr
                sd = 0.06 + 0.1 * litk
                r *= 1.0 - seam * sd * 0.8
                g *= 1.0 - seam * sd
                b *= 1.0 - seam * sd * 0.55
                buf[y, x, 0] += (r - buf[y, x, 0]) * cov
                buf[y, x, 1] += (g - buf[y, x, 1]) * cov
                buf[y, x, 2] += (b - buf[y, x, 2]) * cov


def _offsets(rng, n_shapes=64):
    """unit cluster shapes: a central head plus a ring of smaller heads (flower bunch silhouette)."""
    rows = []
    idx = []
    for s in range(n_shapes):
        ns = int(rng.integers(6, 12))
        idx.append((len(rows), ns))
        rows.append((rng.normal(0, 0.05), rng.normal(0, 0.05), rng.uniform(0.5, 0.62)))
        a0 = rng.uniform(0, 6.283)
        for k in range(ns - 1):
            a = a0 + k * 6.283 / (ns - 1) + rng.normal(0, 0.25)
            d = rng.uniform(0.45, 0.72)
            rows.append((math.cos(a) * d, math.sin(a) * d * 0.85, rng.uniform(0.22, 0.38)))
    return np.array(rows, np.float64), idx


def paint_clusters(col, a, pink, R, light_xy, rng, density=1.0, horizon_row=None):
    """col (h, w, 3) straight colour, a (h, w) alpha, pink (h, w) bool blossom mask, R flower radius px.
    Returns new colour.  Painted at 2x for anti-aliasing."""
    h, w = a.shape
    offs, idx = _offsets(rng)
    lum = col.mean(-1)
    lx, ly = light_xy
    ss = 2
    base = cv2.GaussianBlur(col, (0, 0), max(0.7, R * 0.35))
    cls = []
    # size classes: (radius in flower radii, fraction of area covered)
    for (rc_k, cover) in ((5.0, 0.4), (2.9, 0.45), (1.7, 0.25)):
        Rc = R * rc_k
        cell = Rc * 1.25
        gh, gw = int(h / cell) + 1, int(w / cell) + 1
        jy = ((np.arange(gh)[:, None] + rng.random((gh, gw))) * cell).ravel()
        jx = ((np.arange(gw)[None, :] + rng.random((gh, gw))) * cell).ravel()
        iy = np.clip(jy.astype(int), 0, h - 1)
        ix = np.clip(jx.astype(int), 0, w - 1)
        lk_ = np.clip((lum[iy, ix] - 0.5) / 0.35, 0, 1)
        ok = (a[iy, ix] > 0.9) & pink[iy, ix] & (rng.random(len(iy)) < cover * density * (0.08 + 0.92 * lk_))
        for k in np.nonzero(ok)[0]:
            y, x = iy[k], ix[k]
            c = base[y, x]
            lk = float(np.clip((lum[y, x] - 0.5) / 0.35, 0, 1))
            s = int(rng.integers(len(idx)))
            o0, ns = idx[s]
            rcx = Rc * rng.uniform(0.75, 1.25)
            cls.append((jx[k] * ss, jy[k] * ss, rcx * ss, ns, o0, c[0], c[1], c[2], lk, 1.0 - lk, 0))
    if not cls:
        return col
    cl = np.array(cls, np.float64)
    # big bunches first, smaller ones painted over them; within a class, shaded first then lit on top
    order = np.lexsort((cl[:, 8], -cl[:, 2]))
    cl = cl[order]
    big = cv2.resize(col, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    am = cv2.resize(a, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    pm = cv2.resize(pink.astype(np.float32), (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    am = np.minimum(am, 0.5 + pm).astype(np.float32)
    _stamp(big, am, cl, offs, float(lx), float(ly))
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
