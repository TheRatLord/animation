"""Technique C clouds: volumetric-lite light estimate + painterly paint-over (round 2).

Why not round 1: sphere/SDF unions shaded per lobe read as clay / Michelin balloons with AO dimples.
Here the GEOMETRY is a 2.5D relief of a few big painted MASSES and the SHADING is a posterised
single-scattering estimate, so interiors are flat painted value masses and all the detail sits on
the (crisp, cauliflower) edges.

Pipeline (per plate)
  1. MASSES  (Relief.mass, numba) - cauliflower heads: warped ellipse silhouettes whose edge is
     displaced by multi-scale 2D dome noise (bumps concentrated on the UPPER edge, flat/cut bases).
     Added strictly back-to-front (painter's order): a nearer head always covers what is behind it
     with its own crisp silhouette -> clean overlapping-mass edges, never blended craters.
     Each head also writes a surface height Z (toward the camera) riding on top of the relief
     behind it + a global column bulge (tower) -> a 2.5D height slab |z| < Z(x, y).
  2. LIGHT   - numba ray-march toward the sun through that slab from a few depths under every
     visible surface point (single scattering, transmittance exp(-k * path inside)).  The sea of
     clouds uses the same idea on a world-space heightfield of the cloud tops (long shadows of the
     towers and higher billows fall into the valleys).
  3. PAINT   (paint_relief) - the light is smoothed INSIDE each head only (normalised convolution per
     mass -> big smooth planes, crisp edges), posterised into lit / mid / shade bands whose
     contours are displaced along their normal by the dome noise (scalloped cauliflower
     terminators, no islands), combined with a per-head planar light (lit crowns / shaded
     undersides), recoloured from a hand-picked palette (warm white, lilac mid, cool lavender /
     blue shade with sky-lit crowns), a short shade band on the back mass behind every front edge,
     bilateral + Kuwahara smoothing, extra fine cauliflower only on the OUTER silhouette, crisp
     2-3 px silver/gold rims on sun-facing edges (+ backlit darkening and a thin translucent edge
     glow right at the sun), torn wispy base, haze, speck removal.

Public API (all plates are straight-alpha RGBA float32; sizes in px; resolution independent via scale)
  tower_plate(pw, ph, seed, pal, sun_dir, sun_px, base_row, top_row, cx0, keys, anvil, anvil_dir,
              scale, head_dir=None, return_sun=False, **paint_kw) -> rgba [, sun_px]
      towering cumulus / cumulonimbus. sun_dir = (x right, y up, z toward camera).
      sun_px = sun position in plate px, or ('edge', row, tuck_px) to tuck the sun just behind the
      right silhouette at that row (return_sun=True returns where it went).
  cumulus_plate(pw, ph, seed, pal, sun_dir, sun_px, keys, scale, **kw) -> rgba   plain cumulus
  sea_plates(W, H, horizon, cam_h, fov, seed, sun_px, sun_elev, splits, towers, ...) -> dict(
      layers=[rgba far..near split at `splits` depths], zrow=plane depth per row (parallax),
      f=focal px, project=(X, Y, Z)->(x, y), rgba=full sea)
  premul(rgba), place(dst, pm, x, y, zoom, pivot) -> alpha    compositing helpers
  paint_relief(...) knobs: bands, mid_w, head_t/head_t2, scallop_px, meander_px, rim, rim_px, hdr,
      glow, backlit, tear, haze/haze_top, sil_detail, edge_accent, shade_rim, head_rim, pal_maps.
  Palettes: PAL_SUMMER, PAL_SEA, PAL_TOWER_DAWN.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


# =============================================================================== small helpers

def _hex(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _col(c):
    return _hex(c) if isinstance(c, str) else np.asarray(c, np.float32)


def fblur(img, sigma):
    """Blur whose cost is independent of sigma."""
    if sigma <= 0.3:
        return img
    if sigma <= 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3.0)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


# =============================================================================== numba noise

@njit(cache=True, inline='always')
def _h3(ix, iy, iz, seed):
    n = np.int64(ix) * np.int64(374761393) + np.int64(iy) * np.int64(668265263) \
        + np.int64(iz) * np.int64(1440662683) + np.int64(seed) * np.int64(2654435761)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n ^ (n >> np.int64(16))
    return (n & np.int64(0xFFFFFF)) / 16777216.0


@njit(cache=True, inline='always')
def _vn3(x, y, z, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    uz = fz * fz * (3 - 2 * fz)
    a = _h3(ix, iy, iz, seed)
    b = _h3(ix + 1, iy, iz, seed)
    c = _h3(ix, iy + 1, iz, seed)
    d = _h3(ix + 1, iy + 1, iz, seed)
    e = _h3(ix, iy, iz + 1, seed)
    f = _h3(ix + 1, iy, iz + 1, seed)
    g = _h3(ix, iy + 1, iz + 1, seed)
    h = _h3(ix + 1, iy + 1, iz + 1, seed)
    x1 = a + (b - a) * ux
    x2 = c + (d - c) * ux
    x3 = e + (f - e) * ux
    x4 = g + (h - g) * ux
    y1 = x1 + (x2 - x1) * uy
    y2 = x3 + (x4 - x3) * uy
    return y1 + (y2 - y1) * uz


@njit(cache=True, inline='always')
def _fbm3(x, y, z, seed, oct):
    s = 0.0
    a = 0.5
    t = 0.0
    for o in range(oct):
        s += a * _vn3(x, y, z, seed + o * 17)
        t += a
        a *= 0.5
        x = x * 2.03 + 1.7
        y = y * 2.03 + 3.1
        z = z * 2.03 + 5.3
    return s / t


@njit(cache=True, inline='always')
def _h2(ix, iy, seed):
    return _h3(ix, iy, 7, seed)


@njit(cache=True, inline='always')
def _dome2(x, y, seed, sx):
    """2D dome/cauliflower noise with anisotropic cells (sx > 1 = cells wider than tall)."""
    x = x / sx
    ix = math.floor(x)
    iy = math.floor(y)
    best = 9.0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            cx = ix + dx
            cy = iy + dy
            px = cx + 0.1 + 0.8 * _h2(cx, cy, seed)
            py = cy + 0.1 + 0.8 * _h2(cx, cy, seed + 1)
            rr = 0.55 + 0.45 * _h2(cx, cy, seed + 2)
            d = ((px - x) ** 2 + (py - y) ** 2) / (rr * rr)
            if d < best:
                best = d
    v = 1.0 - best / 0.9
    if v < 0.0:
        v = 0.0
    return math.sqrt(v)


@njit(cache=True, parallel=True, fastmath=True)
def _dome_field(h, w, cell, seed, sx, oy):
    out = np.empty((h, w), np.float32)
    for j in prange(h):
        for i in range(w):
            out[j, i] = _dome2(i / cell, (j + oy) / cell, seed, sx)
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _vn_field(h, w, cx, cy, seed, oct):
    out = np.empty((h, w), np.float32)
    for j in prange(h):
        for i in range(w):
            out[j, i] = _fbm3(i / cx, j / cy, 0.37, seed, oct)
    return out


def dome_noise(h, w, cell, seed=0, sx=1.0):
    return _dome_field(int(h), int(w), float(cell), int(seed), float(sx), 0.0)



# =============================================================================== 2.5D relief slab + light march

@njit(cache=True, inline='always')
def _bil(Z, y, x):
    H, W = Z.shape
    if x < 0 or y < 0 or x > W - 1.001 or y > H - 1.001:
        return 0.0
    i = int(x)
    j = int(y)
    fx = x - i
    fy = y - j
    a = Z[j, i] + (Z[j, i + 1] - Z[j, i]) * fx
    b = Z[j + 1, i] + (Z[j + 1, i + 1] - Z[j + 1, i]) * fx
    return a + (b - a) * fy


@njit(cache=True, parallel=True, fastmath=True)
def _slab_light(Z, sx, sy, sz, k, steps, step0, grow, depths, wts, zmax):
    """Single scattering through the slab |z| < Z(x, y). For every visible surface point, a few
    samples below the surface march toward the sun accumulating the path length inside the slab.
    Returns transmittance-weighted light (H, W)."""
    H, W = Z.shape
    out = np.zeros((H, W), np.float32)
    nd = depths.shape[0]
    for j in prange(H):
        for i in range(W):
            z0 = Z[j, i]
            if z0 <= 0.0:
                continue
            acc = 0.0
            for d in range(nd):
                px = i + 0.0
                py = j + 0.0
                pz = z0 - depths[d]
                if pz < -z0:
                    pz = -z0
                od = 0.0
                st = step0
                for s in range(steps):
                    px += sx * st
                    py -= sy * st
                    pz += sz * st
                    if pz > zmax or px < 0 or py < 0 or px > W - 1 or py > H - 1:
                        break
                    zz = _bil(Z, py, px)
                    if pz < zz and pz > -zz:
                        od += st
                    st *= grow
                acc += wts[d] * math.exp(-k * od)
            out[j, i] = acc
    return out


def _vn2(h, w, cell_x, cell_y, seed, oct=3):
    return _vn_field(int(h), int(w), float(cell_x), float(cell_y), int(seed), int(oct))


@njit(cache=True, fastmath=True)
def _mass_kernel(Z, E, ID, A, G, D1, D2, D3, W1, W2, F, Zg, tagZ, tagID, prm, n, tagval, x0, x1, y0, y1):
    cx, cy, a, b, bb, z0, zd, bump, up_only, warp = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], \
        prm[7], prm[8], prm[9]
    fb, ft, zg, fibrous, lift, head_bias, opacity, edge_soft = prm[10], prm[11], prm[12], prm[13], prm[14], \
        prm[15], prm[16], prm[17]
    c0, c1, c2, scale, s2x, s2y, tzv = prm[18], prm[19], prm[20], prm[21], prm[22], prm[23], prm[24]
    rr = min(a, b)
    szf = min(1.0, rr / (2.2 * c0)) * bump
    aa = (0.9 * scale + 0.25) * edge_soft
    use_tag = tagZ.shape[0] > 1
    for y in range(y0, y1):
        for x in range(x0, x1):
            wx = x + W1[y, x] * warp * 2 * a
            wy = y + W2[y, x] * warp * 2 * b
            u = (wx - cx) / a
            vb = b if wy < cy else bb
            v = (wy - cy) / vb
            r = math.sqrt(u * u + v * v) + 1e-6
            sd = (1.0 - r) * rr
            if sd < -2.5 * c0:
                continue
            up = -v / r
            if up < 0.0:
                up = 0.0
            side = abs(u) / r
            q = up * 1.2 + 0.25 * side
            if q > 1.0:
                q = 1.0
            wgt = (1 - up_only) + up_only * q
            if fibrous > 0:
                bmp = (0.35 * c1 * F[y, x] + 0.1 * c2 * (D3[y, x] - 0.5)) * wgt * bump
            else:
                bmp = (0.8 * c0 * szf * (D1[y, x] - 0.5) + 0.45 * c1 * (D2[y, x] - 0.5)
                       + 0.18 * c2 * (D3[y, x] - 0.5)) * wgt * bump
            sdm = sd + bmp
            if not math.isnan(fb):
                sdm = min(sdm, fb - y)
            if not math.isnan(ft):
                sdm = min(sdm, (y - ft) + 0.2 * c2 * (D3[y, x] - 0.5))
            al = 0.5 + sdm / aa
            if al <= 0.0:
                continue
            if al > 1.0:
                al = 1.0
            t = sdm / (0.6 * rr)
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            sh = 1.0 - (1.0 - t) ** 2.2
            zs = z0 + zd * sh + Zg[y, x] * zg + lift * (cy - y) / max(b, 1.0)
            zs = max(zs, Z[y, x] + zd * sh * 0.6 + 1.0)
            if al > 0.5:
                Z[y, x] = zs
                E[y, x] = sdm
                ID[y, x] = n
                g = (u * s2x - v * s2y) * 0.8 + head_bias
                G[y, x] = min(max(g, -1.0), 1.0)
                if use_tag:
                    tagZ[y, x] = tzv
                    tagID[y, x] = tagval
            ao = al * opacity
            if ao > A[y, x]:
                A[y, x] = ao


class Relief:
    """Accumulates cloud MASSES into a 2.5D relief: Z (front surface toward camera, px), alpha,
    E (distance to the winning mass's own edge, px), id (winning mass), hf (height fraction)."""

    def __init__(self, pw, ph, seed=0, scale=1.0, cells=(95, 42, 17), sun_dir=(0.7, 0.5, 0.4)):
        self.pw, self.ph, self.seed, self.scale = pw, ph, seed, scale
        self.Z = np.zeros((ph, pw), np.float32)
        self.A = np.zeros((ph, pw), np.float32)
        self.E = np.full((ph, pw), 1e3, np.float32)
        self.ID = np.full((ph, pw), -1, np.int32)
        self.Zg = None
        cs = [c * scale for c in cells]
        self.D = [dome_noise(ph, pw, cs[0], seed + 1, 1.3), dome_noise(ph, pw, cs[1], seed + 2, 1.2),
                  dome_noise(ph, pw, cs[2], seed + 3, 1.0)]
        self.cells = cs
        self.W1 = _vn2(ph, pw, 160 * scale, 160 * scale, seed + 5, 3) - 0.5
        self.W2 = _vn2(ph, pw, 160 * scale, 160 * scale, seed + 6, 3) - 0.5
        self.F = _vn2(ph, pw, 150 * scale, 16 * scale, seed + 7, 3) - 0.5     # fibrous (anvil) noise
        self.G = np.zeros((ph, pw), np.float32)                                 # per-head planar light
        sd = np.asarray(sun_dir, np.float64)
        n = math.hypot(sd[0], sd[1]) + 1e-6
        self.sun2 = (sd[0] / n, sd[1] / n)
        self.n = 0
        self.anvil_id = -1

    def mass(self, cx, cy, a, b, bb=None, z0=0.0, zd=None, bump=1.0, up_only=0.85, warp=0.05, flat_bottom=None,
             zg=1.0, stretch=1.0, fibrous=0.0, lift=0.0, flat_top=None, head_bias=0.0, opacity=1.0, edge_soft=1.0,
             tagZ=None, tagZval=0.0, tagID=None, tagval=0):
        """Add one cauliflower head centred at (cx, cy) with half-width a, top half-height b and
        bottom half-height bb (px). z0: depth offset (masses must be added back to front).
        zd: thickness bulge. bump: cauliflower amplitude; up_only: 0 = bumps all round, 1 = only on the
        upper edge; flat_bottom / flat_top: rows where the mass is cut flat; fibrous: >0 = stretched,
        smoother edge (anvil). tagZ/tagID: optional extra (H, W) arrays set to tagZval/tagval where
        this mass wins. Returns True if the mass touched the plate."""
        pw, ph = self.pw, self.ph
        bb = b * 1.25 if bb is None else bb
        zd = 0.55 * min(a, b) if zd is None else zd
        m = 1.3
        pad = 4 + 0.5 * self.cells[0] * bump
        x0, x1 = int(max(cx - a * m - pad, 0)), int(min(cx + a * m + pad, pw))
        y0, y1 = int(max(cy - b * m - pad, 0)), int(min(cy + bb * m + pad, ph))
        if x1 <= x0 or y1 <= y0:
            return False
        if self.Zg is None:
            self.Zg = np.zeros((ph, pw), np.float32)
        if tagZ is None:
            tagZ = np.zeros((1, 1), np.float32)
            tagID = np.zeros((1, 1), np.int32)
        prm = np.array([cx, cy, a, b, bb, z0, zd, bump, up_only, warp,
                        np.nan if flat_bottom is None else flat_bottom, np.nan if flat_top is None else flat_top,
                        zg, fibrous, lift, head_bias, opacity, edge_soft, self.cells[0], self.cells[1],
                        self.cells[2], self.scale, self.sun2[0], self.sun2[1], tagZval], np.float64)
        _mass_kernel(self.Z, self.E, self.ID, self.A, self.G, self.D[0], self.D[1], self.D[2], self.W1, self.W2,
                     self.F, self.Zg, tagZ, tagID, prm, int(self.n), int(tagval), x0, x1, y0, y1)
        self.n += 1
        return True


def column_Zg(pw, ph, rows_cx_R, zscale=1.0):
    """Global cylinder bulge of a tower: rows_cx_R = (cx(row), R(row)) arrays of length ph."""
    cx, R = rows_cx_R
    xs = np.arange(pw, dtype=np.float32)[None, :]
    Rw = np.maximum(R[:, None], 1) * 1.6
    u = (xs - cx[:, None]) / Rw
    return (np.clip(1 - u * u, 0, 1) ** 1.5 * Rw * zscale).astype(np.float32)


# =============================================================================== layouts

def tower_layout(pw, ph, seed=0, base_row=None, top_row=None, cx0=None, keys=None, anvil=True, scale=1.0,
                 anvil_dir=1.0, crown=1.0, n_inner=3):
    """Mass list for a towering cumulus / cumulonimbus. keys: (height fraction 0..1, half-width as
    fraction of the tower height, centre offset as fraction of the height).
    Silhouette heads run up both flanks; a few big interior masses give the crisp internal edges;
    lower masses are nearer the camera (we look up at the tower). Returns (masses, (C(row), R(row)), anvil, zspan)."""
    rng = np.random.default_rng(seed)
    base_row = ph * 0.9 if base_row is None else base_row
    top_row = ph * 0.08 if top_row is None else top_row
    cx0 = pw * 0.45 if cx0 is None else cx0
    Ht = base_row - top_row
    if keys is None:
        keys = [(0.0, 0.46, -0.02), (0.12, 0.48, -0.02), (0.3, 0.38, 0.02), (0.5, 0.30, 0.06),
                (0.68, 0.26, 0.09), (0.82, 0.25, 0.12), (0.93, 0.2, 0.14), (1.0, 0.1, 0.15)]
    k = np.array(keys, np.float64)

    def prof(f):
        return np.interp(f, k[:, 0], k[:, 1]) * Ht, cx0 + np.interp(f, k[:, 0], k[:, 2]) * Ht
    rows = np.arange(ph, dtype=np.float32)
    f_rows = np.clip((base_row - rows) / Ht, 0, 1)
    Rr, Cr = prof(f_rows)
    masses = []
    zspan = 0.9 * Ht * float(k[:, 1].max())

    def zof(f):
        return (1 - f) * zspan * 0.55 + rng.uniform(0, 0.06) * zspan
    # flank heads (the silhouette)
    for side in (-1, 1):
        f = rng.uniform(0.0, 0.05)
        while f < 0.86:
            R, C = (float(v) for v in prof(f))
            a = R * rng.uniform(0.38, 0.55)
            b = a * rng.uniform(0.55, 0.8)
            cx = C + side * (R - a * rng.uniform(0.6, 0.85))
            cy = base_row - f * Ht - b * 0.3
            masses.append(dict(cx=cx, cy=cy, a=a, b=b, bb=b * 1.5, z0=zof(f), zd=0.5 * min(a, b),
                               flat_bottom=base_row if f < 0.12 else None))
            f += rng.uniform(0.15, 0.24)
    # core body: broad masses along the axis, set back (fill gaps, no visible edges of their own)
    f = 0.0
    while f < 0.9:
        R, C = (float(v) for v in prof(f))
        a = R * 0.95
        b = 0.1 * Ht
        masses.append(dict(cx=C, cy=base_row - f * Ht - 0.5 * b, a=a, b=b, bb=b * 1.5,
                           z0=zof(f) - 0.15 * zspan, zd=0.4 * b, bump=0.5, flat_bottom=base_row if f < 0.1 else None))
        f += 0.08
    # big interior masses: broad planes whose upper edges read as crisp overlapping edges
    fs = np.sort(rng.uniform(0.0, 0.78, n_inner))
    for f in fs:
        R, C = (float(v) for v in prof(f))
        a = R * rng.uniform(0.6, 0.95)
        b = a * rng.uniform(0.55, 0.75)
        cx = C + rng.uniform(-0.35, 0.35) * R
        cy = base_row - f * Ht - b * 0.2
        masses.append(dict(cx=cx, cy=cy, a=a, b=b, bb=b * 1.6, z0=zof(f) + 0.06 * zspan, zd=0.45 * min(a, b),
                           flat_bottom=base_row if f < 0.15 else None))
    # crown: a few big heads
    R, C = (float(v) for v in prof(0.88))
    for xf in [-0.45, 0.2, 0.6][:max(1, int(round(3 * crown)))]:
        a = R * rng.uniform(0.6, 0.8)
        b = a * rng.uniform(0.6, 0.8)
        masses.append(dict(cx=C + xf * R, cy=base_row - 0.9 * Ht - b * 0.3 + abs(xf) * b * 0.7, a=a, b=b,
                           bb=b * 1.8, z0=0.1 * zspan * rng.uniform(0.3, 1.0), zd=0.5 * min(a, b)))
    an = None
    if anvil:
        R, C = (float(v) for v in prof(0.9))
        an = dict(y_top=base_row - 0.985 * Ht, y_bot=base_row - 0.72 * Ht, x_root=float(C),
                  x_tip=float(C) + anvil_dir * 1.45 * float(k[:, 1].max()) * Ht,
                  x_back=float(C) - anvil_dir * 0.3 * float(R))
    return masses, (Cr.astype(np.float32), Rr.astype(np.float32)), an, zspan


def _anvil(Rf, an, zspan, scale):
    """Anvil: ONE flat-topped wedge spreading downwind (thick at the crown, thin at the tip), smooth
    slightly domed top, fibrous underside and tip, thinning to translucency at the tip."""
    pw, ph = Rf.pw, Rf.ph
    yt, yb = an['y_top'], an['y_bot']
    xr, xt = an['x_back'], an['x_tip']
    th0 = yb - yt
    L = xt - xr
    x0, x1 = int(max(min(xr, xt) - 40, 0)), int(min(max(xr, xt) + 40, pw))
    y0, y1 = int(max(yt - 40, 0)), int(min(yb + 60, ph))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    sl = (slice(y0, y1), slice(x0, x1))
    t = (xs - xr) / L                                         # 0 at the root, 1 at the tip
    tc = np.clip(t, 0, 1)
    top = yt + th0 * 0.08 * tc ** 1.5 - th0 * 0.05 * np.sin(np.pi * tc)
    th = th0 * (1.0 - 0.85 * tc ** 0.8)
    F = Rf.F[sl]
    D2, D3 = Rf.D[1][sl], Rf.D[2][sl]
    s_top = (ys - top) + 0.1 * Rf.cells[1] * (Rf.D[0][sl] - 0.5) + 0.04 * Rf.cells[1] * (D2 - 0.5)
    s_bot = (top + th - ys) + 0.28 * Rf.cells[1] * F * (0.4 + tc) + 0.12 * Rf.cells[2] * (D3 - 0.5)
    s_tip = (1.0 - t) * abs(L) * 0.25 + 0.6 * Rf.cells[1] * F
    s_root = t * abs(L) + 0.3 * th0
    sdm = np.minimum(np.minimum(s_top, s_bot), np.minimum(s_tip, s_root))
    soft = (0.9 * Rf.scale + 0.25) * (1.0 + 4.0 * tc ** 2)
    al = np.clip(0.5 + sdm / soft, 0, 1) * (1.0 - 0.45 * tc ** 2)
    v = np.clip((ys - top) / np.maximum(th, 1), 0, 1)
    zs = zspan * 0.02 + 0.3 * th0 * np.clip(sdm / (0.4 * th0), 0, 1)
    win = (al > 0.3) & (zs > Rf.Z[sl])
    Rf.Z[sl][win] = zs[win]
    Rf.E[sl][win] = sdm[win]
    Rf.ID[sl][win] = Rf.n
    g = np.clip(0.95 - 1.7 * v + 0.2 * (1 - tc), -1, 1)       # lit top, shaded underside
    Rf.G[sl][win] = g[win]
    np.maximum(Rf.A[sl], al, out=Rf.A[sl])
    Rf.anvil_id = Rf.n
    Rf.n += 1


# =============================================================================== palettes

PAL_SUMMER = dict(hi='#f9f9f7', lit='#f1f0ee', mid='#cdcdec', shd_hi='#9ba9e2', shd_lo='#6f7dc8',
                  bounce='#b4bbe8', base='#8e95cc', rim='#fffbf0', haze='#cfe6f8', glow='#fff4dc')


def _pal(p):
    return {k: _col(v) for k, v in p.items()}


def _lerp(a, b, t):
    if np.ndim(t) == 2:
        t = t[..., None]
    return a + (b - a) * t


def kuwahara(img, r=3):
    """Fast (box-filter) Kuwahara on a float32 (H,W,3) image: flattens texture into painted patches
    while keeping edges."""
    k = r + 1
    lum = img.mean(-1)
    mean_c = cv2.blur(img, (k, k))
    mean_l = cv2.blur(lum, (k, k))
    var = cv2.blur(lum * lum, (k, k)) - mean_l * mean_l
    H, W = lum.shape
    best = np.full((H, W), 1e9, np.float32)
    out = np.zeros_like(img)
    o = r // 2
    for dy, dx in ((-o, -o), (-o, o), (o, -o), (o, o)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        v = cv2.warpAffine(var, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(mean_c, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        sel = v < best
        best = np.where(sel, v, best)
        out = np.where(sel[..., None], m, out)
    return out


# =============================================================================== paint-over

def smooth_by_mass(L, ID, sigma, weight=None):
    """Normalised-convolution blur of L inside each mass separately: interiors become broad smooth
    value masses while the edges between overlapping masses stay crisp. Small masses get their
    weighted mean, i.e. one flat painted value."""
    from scipy import ndimage
    Lf = L.astype(np.float32)
    w = np.ones_like(Lf) if weight is None else (weight + 1e-3).astype(np.float32)
    lab = ID + 1
    nlab = int(lab.max())
    if nlab <= 0:
        return np.zeros_like(Lf)
    idx = np.arange(1, nlab + 1)
    sums = ndimage.sum_labels(Lf * w, lab, idx)
    wts = ndimage.sum_labels(w, lab, idx)
    means = np.concatenate([[0.0], sums / np.maximum(wts, 1e-6)]).astype(np.float32)
    out = means[lab]
    objs = ndimage.find_objects(lab)
    pad = int(3 * sigma) + 2
    for i, so in enumerate(objs):
        if so is None:
            continue
        ys, xs = so
        if (ys.stop - ys.start) < 1.5 * sigma and (xs.stop - xs.start) < 1.5 * sigma:
            continue
        y0, y1 = max(ys.start - pad, 0), min(ys.stop + pad, L.shape[0])
        x0, x1 = max(xs.start - pad, 0), min(xs.stop + pad, L.shape[1])
        mm = (lab[y0:y1, x0:x1] == i + 1).astype(np.float32)
        ww = mm * w[y0:y1, x0:x1]
        num = fblur(Lf[y0:y1, x0:x1] * ww, sigma)
        den = fblur(ww, sigma)
        sm = num / np.maximum(den, 1e-4)
        o = out[y0:y1, x0:x1]
        o[mm > 0] = sm[mm > 0]
    return out


def light_relief(Rf, sun_dir, k=None, depths=(0.0, 8.0, 20.0, 40.0), wts=(0.35, 0.3, 0.2, 0.15), steps=48):
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / np.linalg.norm(sd)
    sc = Rf.scale
    k = (1.0 / (38.0 * sc)) if k is None else k
    dep = np.asarray(depths, np.float64) * sc
    return _slab_light(np.ascontiguousarray(Rf.Z), sd[0], sd[1], sd[2], k, steps, 2.0 * sc, 1.07,
                       dep, np.asarray(wts, np.float64), float(Rf.Z.max()) + 1)


def paint_relief(Rf, L, pal=PAL_SUMMER, sun_dir=(0.6, 0.5, 0.35), sun_px=None, base_row=None, top_row=None,
                 bands=(0.45,), band_px=1.5, mid_w=0.07, head_t=0.46, head_t2=0.62, rim=1.0, rim_px=2.5, tear=1.0, glow=1.0, hdr=1.5,
                 lit_bias=0.0, shade_light=0.0, edge_accent=1.0, haze=0.0, haze_top=0.6, seed=0, smooth_px=40.0, scallop_px=22.0, sil_detail=1.0, backlit=0.06, abs_bands=False,
                 pal_maps=None, amb_z=0.45, shade_rim=0.0, head_rim=0.0, meander_px=40.0, speck_px=500.0):
    """Posterise + paint the relief. Returns RGBA (ph, pw, 4) float32, straight alpha.
    pal_maps: optional {key: (H, W, 3) array} overriding palette colours per pixel (e.g. lit tops gold
    near the sun, pink away from it)."""
    P = _pal(pal)
    if pal_maps:
        P.update({k: np.asarray(v, np.float32) for k, v in pal_maps.items()})
    ph, pw, sc = Rf.ph, Rf.pw, Rf.scale
    alpha = Rf.A.copy()
    inside = alpha > 0.5
    rows = np.arange(ph, dtype=np.float32)[:, None]
    base_row = ph if base_row is None else base_row
    top_row = 0 if top_row is None else top_row
    # --- underpainting: edge-preserving smoothing of the light (kills speckle, keeps mass planes)
    Ls = smooth_by_mass(L, Rf.ID, smooth_px * sc, _ss(6 * sc, 16 * sc, Rf.E))
    if inside.any() and not abs_bands:
        lo, hi_ = np.percentile(Ls[inside], [2, 98])
        Ls = np.clip((Ls - lo) / (hi_ - lo + 1e-6), 0, 1)
    Gh = 0.5 + 0.5 * Rf.G
    D1, D2, D3 = Rf.D
    noise = 0.55 * (D2 - 0.5) + 0.3 * (D1 - 0.5) + 0.15 * (D3 - 0.5)

    def band(F, q, disp):
        """Crisp anti-aliased band mask of field F at quantile/threshold q with its contour displaced
        by up to `disp` px along the normal by the cauliflower noise (-> scalloped, no islands)."""
        Fb = fblur(F, 3.0 * sc)
        gy_, gx_ = np.gradient(Fb)
        gm = np.sqrt(gx_ * gx_ + gy_ * gy_)
        if inside.any():
            g40, g80 = np.percentile(gm[inside], [40, 80])
            gm = np.clip(gm, g40, g80)
        Fp = F + gm * sc * (disp * noise * 2 + meander_px * Rf.W1 * 2)
        return Fp, gm

    # global light (volumetric) -> which parts of the tower face the sun
    Lp, gmL = band(Ls, 0, scallop_px * 1.3)
    if abs_bands:
        t_g = bands[0] - lit_bias
    else:
        t_g = np.quantile(Lp[inside][::5], bands[0]) - lit_bias if inside.any() else 0.5
    w = gmL * sc * band_px + 1e-4
    k_lit = _ss(t_g - w, t_g + w, Lp)
    t_m = t_g - mid_w
    k_mid = _ss(t_m - w, t_m + w, Lp)
    # per-head planar light -> lit caps / shaded undersides inside both regions
    Gp, gmG = band(Gh, 0, scallop_px)
    wg = gmG * sc * band_px + 1e-4
    h_lit = _ss(head_t - wg, head_t + wg, Gp)                  # inside the lit side: most is lit
    h_shd = _ss(head_t2 - wg, head_t2 + wg, Gp)                # inside the shade: lighter crowns
    # ambient (sky light): height in the cloud + upward facing
    hf = np.clip((base_row - rows) / max(base_row - top_row, 1), 0, 1) * np.ones((1, pw), np.float32)
    Zb = fblur(Rf.Z, 25 * sc)
    gy = np.gradient(Zb, axis=0)
    upf = np.clip(0.5 + gy * 1.5, 0, 1)
    amb = np.clip((1 - amb_z) * hf + amb_z * upf + shade_light, 0, 1)
    shade = _lerp(P['shd_lo'], P['shd_hi'], amb)
    nearb = _ss(base_row - 0.2 * (base_row - top_row), base_row, rows) * np.ones((1, pw), np.float32)
    shade = _lerp(shade, P['bounce'], 0.3 * nearb)
    shade = _lerp(shade, _lerp(P['shd_hi'], P['mid'], 0.45), h_shd * 0.8)
    mid = P['mid'] * np.ones_like(shade)
    lit = _lerp(P['lit'], P['hi'], _ss(0.6, 0.95, Gp) * _ss(t_g, t_g + 0.3, Lp))
    lit = _lerp(_lerp(P['lit'], P['mid'], 0.35), lit, _ss(0.0, 0.6, hf))       # lower lit masses a touch greyer
    lit_side = _lerp(mid, lit, h_lit)
    col = _lerp(shade, mid, k_mid)
    col = _lerp(col, lit_side, k_lit)
    col = _lerp(col, P['base'], 0.5 * _ss(base_row - 30 * sc, base_row, rows) * (1 - k_lit))
    if shade_rim > 0:
        # sky-lit thin lighter rim on the upper edge of every head inside the shade
        tr = _ss(4.0 * sc, 1.5 * sc, Rf.E) * _ss(0.55, 0.8, Gh) * (1 - k_mid) * inside
        col = _lerp(col, _lerp(P['shd_hi'], P['mid'], 0.5), np.clip(tr * shade_rim, 0, 1))
    if head_rim > 0:
        # backlit silver/gold lining on the upper edge of EVERY head (internal edges included)
        from scipy import ndimage
        lab = Rf.ID + 1
        emax = ndimage.maximum(Rf.E, lab, np.arange(0, int(lab.max()) + 1))
        emax = np.asarray(emax, np.float32)[lab]                      # thickness of each head (px)
        hr = _ss((rim_px + 1.0) * sc, (rim_px - 0.8) * sc, Rf.E) * _ss(0.6, 0.85, Gh) * inside * k_mid
        hr = hr * _ss(8 * sc, 22 * sc, emax)                          # no rings around tiny heads
        col = _lerp(col, _lerp(P['lit'], P['rim'], 0.6) * hdr, np.clip(hr * head_rim, 0, 1))
    k_lit = k_lit * h_lit
    col = col.astype(np.float32)
    # --- painterly smoothing
    col = cv2.bilateralFilter(col, int(7 * sc) | 1, 0.06, 3.0 * sc)
    col = kuwahara(col, max(2, int(3 * sc)))
    # --- overlapping-mass edges: front mass edge gets a thin light lip; the mass behind a soft
    #     darker contact band (only a few px, only in the lit/mid bands -> no AO craters)
    if edge_accent > 0:
        # masses are added back-to-front, so a larger ID is nearer. A pixel whose neighbourhood holds a
        # larger ID sits on a back mass just behind a front mass's edge -> paint a short shade band there
        # (crisp on the front edge side, fading away from it): the painted 'overlap' edge, no AO craters.
        idf = Rf.ID.astype(np.float32)
        behind = np.zeros_like(alpha)
        wsum = 0.0
        for r, wgt_ in ((2, 1.0), (4, 0.9), (7, 0.7), (11, 0.45), (16, 0.25)):
            k_ = int(2 * r * sc) | 1
            mx = cv2.dilate(idf, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_, k_)))
            behind += (mx > idf) * wgt_
            wsum += wgt_
        behind = fblur(behind / wsum, 1.2 * sc) * inside
        tgt = _lerp(mid, shade, 0.3)
        col = _lerp(col, tgt, np.clip(behind * edge_accent * 0.55 * k_mid, 0, 1))
        col = _lerp(col, _lerp(shade, P['shd_lo'], 0.5), np.clip(behind * edge_accent * 0.35 * (1 - k_mid), 0, 1))
    # --- extra fine cauliflower detail concentrated on the OUTER silhouette only
    if sil_detail > 0:
        m0 = (alpha > 0.5).astype(np.uint8)
        sd0 = cv2.distanceTransform(m0, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m0, cv2.DIST_L2, 5)
        sdb0 = fblur(sd0.astype(np.float32), 5.0 * sc)
        gy0, gx0 = np.gradient(sdb0)
        gn0 = np.sqrt(gx0 * gx0 + gy0 * gy0) + 1e-6
        wup = np.clip(0.45 + 0.8 * (gy0 / gn0), 0.15, 1.0)          # gy>0 inside-down -> edge faces up
        fine = (0.55 * Rf.cells[2] * (D3 - 0.5) + 0.3 * Rf.cells[1] * (D2 - 0.5)) * sil_detail * wup
        fine = fine * (1 - nearb)
        if Rf.anvil_id >= 0:
            am = cv2.dilate((Rf.ID == Rf.anvil_id).astype(np.uint8), np.ones((int(15 * sc) | 1,) * 2, np.uint8))
            fine = fine * (1 - 0.9 * am)
        a_new = np.clip(0.5 + (sd0 + np.minimum(fine, 8.0 * sc)) / (0.9 * sc + 0.25), 0, 1)
        near_e = np.abs(sd0) < 14 * sc
        grow = near_e & (alpha < 0.5) & (a_new > 0)
        if grow.any():
            aw = fblur(alpha * (alpha > 0.5), 3.0 * sc)
            cext = fblur(col * (alpha > 0.5)[..., None], 3.0 * sc) / np.maximum(aw, 1e-4)[..., None]
            col = np.where(grow[..., None], cext, col).astype(np.float32)
        alpha = np.where(near_e, a_new, alpha).astype(np.float32)
        inside = alpha > 0.5
    # --- silver lining on the outer silhouette + translucent glow near the sun
    sdir = np.asarray(sun_dir, np.float32)
    sx2, sy2 = sdir[0], -sdir[1]
    n2 = math.hypot(sx2, sy2) + 1e-6
    m0 = inside.astype(np.uint8)
    din = cv2.distanceTransform(m0, cv2.DIST_L2, 5)
    sdb = fblur(din - cv2.distanceTransform(1 - m0, cv2.DIST_L2, 5), 4.0 * sc)
    gy2, gx2 = np.gradient(sdb)
    gn = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-6
    nx_, ny_ = -gx2 / gn, -gy2 / gn
    facing = np.clip((nx_ * sx2 + ny_ * sy2) / n2, 0, 1)
    # crisp rim strip: full strength for the first rim_px pixels, 1px anti-aliased falloff
    strip = _ss((rim_px + 1.0) * sc, (rim_px - 1.0) * sc, din) * alpha
    near = np.zeros_like(alpha)
    if sun_px is not None:
        S_ = max(pw, ph)
        xs = np.arange(pw, dtype=np.float32)[None, :]
        dsun = np.sqrt((xs - sun_px[0]) ** 2 + (rows - sun_px[1]) ** 2)
        near = np.exp(-dsun / (0.12 * S_)) + 0.8 * np.exp(-dsun / (0.035 * S_))
        # backlit zone: the cloud body right in front of the sun is in its own shadow (cooler, darker)
        bz = np.exp(-dsun / (backlit * S_)) * (1 - np.exp(-din / (8 * sc))) if backlit > 0 else 0 * din
        bcol = _lerp(_lerp(P['mid'], P['shd_hi'], 0.45), P['glow'], 0.12)
        col = _lerp(col, bcol, np.clip(bz * 0.9 * glow, 0, 0.9))
        # thin translucent edge glowing near the sun (a few px only -> no fuzzy halo)
        te = np.exp(-din / (5 * sc)) * alpha * np.exp(-dsun / (0.06 * S_)) * glow
        col = col + te[..., None] * P['glow'] * 0.5
    rimw = strip * np.clip((0.1 + 0.9 * facing ** 1.5) * (0.35 + 1.6 * near), 0, 1.6) * rim
    # thin cool-white translucent edge on the shadow side (sky light through the thin edge)
    sh_edge = _ss(3.5 * sc, 1.5 * sc, din) * alpha * (1 - k_lit) * (1 - nearb) * rim
    col = _lerp(col, _lerp(P['mid'], P['hi'], 0.35), np.clip(sh_edge * 0.5, 0, 1))
    rimw = np.clip(rimw * (1 - 0.8 * nearb), 0, 1.6)
    rim_col = P['rim'] * (1.0 + (hdr - 1.0) * np.clip(near, 0, 1))[..., None]
    col = _lerp(col, rim_col, np.clip(rimw * 0.9, 0, 1))
    # --- torn wispy base
    if tear > 0 and base_row < ph:
        n1 = _vn2(ph, pw, 55 * sc, 4.5 * sc, seed + 21, 3)
        n2 = _vn2(ph, pw, 150 * sc, 12 * sc, seed + 22, 3)
        tn = 0.55 * n1 + 0.45 * n2
        band = 38 * sc
        dz = np.clip((rows - (base_row - band)) / band, 0, 1.5)
        thr = 0.15 + 0.55 * dz * tear
        keep = _ss(thr - 0.08, thr + 0.08, tn + 0.08 * (D2 - 0.5))
        alpha = alpha * np.where(rows > base_row - band, keep, 1.0)
        foot = np.repeat((Rf.A * (rows < base_row)).max(0, keepdims=True), ph, 0)
        foot = fblur(foot.astype(np.float32), 10 * sc)
        below = np.clip((rows - base_row) / (60 * sc), 0, 1)
        wisp = foot * _ss(0.5, 0.8, tn) * (1 - below) ** 1.5 * (rows > base_row - 12 * sc) * 0.5 * tear
        alpha = np.maximum(alpha, wisp.astype(np.float32))
        col = _lerp(col, P['base'] * 1.05, np.clip(wisp * 1.5, 0, 1) * (rows > base_row))
    if haze > 0:
        hz = _ss(haze_top * ph, ph, rows) * haze * np.ones((1, pw), np.float32)
        col = _lerp(col, P['haze'], hz)
    # drop stray specks (tiny disconnected islands) - a painter would never leave them
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.35).astype(np.uint8), connectivity=8)
    if nlab > 2:
        small = stats[:, cv2.CC_STAT_AREA] < speck_px * sc * sc
        small[0] = False
        if small.any():
            kill = cv2.dilate(small[lab].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            alpha = np.where(kill & ~(alpha > 0.35) | small[lab], 0.0, alpha).astype(np.float32)
    return np.dstack([np.clip(col, 0, None), np.clip(alpha, 0, 1)]).astype(np.float32)


def tower_plate(pw, ph, seed=0, pal=PAL_SUMMER, sun_dir=(0.7, 0.5, 0.45), sun_px=None, base_row=None,
                top_row=None, cx0=None, keys=None, anvil=True, anvil_dir=1.0, scale=None, crown=1.0,
                debug=False, return_sun=False, head_dir=None, **paint_kw):
    """Painted towering cumulus / cumulonimbus plate (RGBA float32 straight alpha, ph x pw).
    sun_dir: (x right, y up, z toward camera). sun_px: sun position in plate px (rim/glow), may be off-plate.
    base_row/top_row: rows of the flat base / top of the tower. keys: tower profile (see tower_layout)."""
    if scale is None:
        scale = ph / 1000.0
    base_row = ph * 0.9 if base_row is None else base_row
    top_row = ph * 0.06 if top_row is None else top_row
    masses, (Cr, Rr), an, zspan = tower_layout(pw, ph, seed, base_row, top_row, cx0, keys, anvil, scale,
                                               anvil_dir, crown)
    Rf = Relief(pw, ph, seed, scale, sun_dir=sun_dir)
    if head_dir is not None:
        n_ = math.hypot(head_dir[0], head_dir[1]) + 1e-6
        Rf.sun2 = (head_dir[0] / n_, head_dir[1] / n_)
    Rf.Zg = column_Zg(pw, ph, (Cr, Rr), 0.8)
    if an:
        _anvil(Rf, an, zspan, scale)
    for m in sorted(masses, key=lambda m: m['z0']):
        Rf.mass(**m)
    if sun_px is not None and sun_px[0] == 'edge':
        # put the sun just behind the right silhouette at row sun_px[1] (tucked in by sun_px[2] px)
        r = int(np.clip(sun_px[1], 0, ph - 1))
        band_ = Rf.A[max(r - 3, 0):r + 4].max(0)
        cols = np.nonzero(band_ > 0.5)[0]
        x = float(cols.max()) if len(cols) else pw * 0.5
        sun_px = (x - sun_px[2], float(r))
    L = light_relief(Rf, sun_dir)
    out = paint_relief(Rf, L, pal, sun_dir, sun_px, base_row, top_row, seed=seed, **paint_kw)
    if debug:
        return out, Rf, L
    if return_sun:
        return out, sun_px
    return out


CU_KEYS = [(0.0, 0.62, 0.0), (0.2, 0.66, 0.0), (0.45, 0.55, 0.03), (0.7, 0.42, 0.06), (0.9, 0.28, 0.08),
           (1.0, 0.12, 0.08)]


def cumulus_plate(pw, ph, seed=0, pal=PAL_SUMMER, sun_dir=(0.7, 0.5, 0.45), sun_px=None, keys=None,
                  scale=None, **kw):
    """A plain (fair-weather / small towering) cumulus filling a pw x ph plate: flat base at 0.86 ph."""
    keys = CU_KEYS if keys is None else keys
    return tower_plate(pw, ph, seed=seed, pal=pal, sun_dir=sun_dir, sun_px=sun_px, base_row=ph * 0.86,
                       top_row=ph * 0.08, cx0=pw * 0.48, keys=keys, anvil=False, scale=scale, crown=0.7, **kw)


# =============================================================================== compositing helpers

def premul(rgba):
    """Straight -> premultiplied RGBA (warp premultiplied plates to avoid dark/colour fringes)."""
    out = rgba.copy()
    out[..., :3] *= out[..., 3:4]
    return np.ascontiguousarray(out, np.float32)


def place(dst, pm, x, y, zoom=1.0, pivot=None, opacity=1.0):
    """Composite premultiplied plate `pm` onto dst (H, W, 3) with its top-left at (x, y) (sub-pixel),
    optionally zoomed about `pivot` (frame coords). Returns the placed alpha (H, W) (for occlusion)."""
    H, W = dst.shape[:2]
    M = np.array([[zoom, 0, x * zoom], [0, zoom, y * zoom]], np.float64)
    if pivot is not None:
        px, py = pivot
        M[0, 2] = px + (x - px) * zoom
        M[1, 2] = py + (y - py) * zoom
    w = cv2.warpAffine(pm, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
    a = w[..., 3:4] * opacity
    dst *= (1 - a)
    dst += w[..., :3] * opacity
    return a[..., 0]


# =============================================================================== sea of clouds (from above)

PAL_SEA = dict(hi='#f9e8cf', lit='#f6bf8c', mid='#e98aa6', shd_hi='#6f5cb4', shd_lo='#2e2672',
               bounce='#b07ab8', base='#4a3c8e', rim='#fff2cf', haze='#f7c0b0', glow='#ffe2a8')
PAL_TOWER_DAWN = dict(hi='#f9e2c2', lit='#f5be8c', mid='#ee8fa8', shd_hi='#9a8ad0', shd_lo='#5b4d9e',
                      bounce='#c48cba', base='#6a5aa6', rim='#fff1d0', haze='#f6c6b8', glow='#ffe6b4')


@njit(cache=True, parallel=True, fastmath=True)
def _hf_light(HF, gx0, gz0, gd, PX, PY, PZ, M, lx, ly, lz, steps, st0, grow, k, depths):
    """Single scattering through a cloud-top heightfield: from each visible surface point (and a few
    points just below it) march toward the sun, accumulating the path length spent below the cloud
    tops (inside the cloud layer). Returns transmittance (H, W)."""
    H, W = PX.shape
    out = np.zeros((H, W), np.float32)
    nz, nx = HF.shape
    for j in prange(H):
        for i in range(W):
            if M[j, i] == 0:
                continue
            acc = 0.0
            for d in range(depths.shape[0]):
                x = PX[j, i]
                y = PY[j, i] - depths[d]
                z = PZ[j, i]
                od = 0.0
                st = st0
                for s in range(steps):
                    x += lx * st
                    y += ly * st
                    z += lz * st
                    gi = (x - gx0) / gd
                    gk = (z - gz0) / gd
                    if gi < 0 or gk < 0 or gi > nx - 1.001 or gk > nz - 1.001:
                        break
                    h = _bil(HF, gk, gi)
                    if y < h:
                        od += st
                    elif y > 6.0:
                        break
                    st *= grow
                acc += math.exp(-k * od)
            out[j, i] = acc / depths.shape[0]
    return out


def sea_plates(W, H, horizon=0.4, cam_h=1.0, fov=52.0, seed=0, sun_px=None, sun_elev=5.0, splits=(),
               pal=PAL_SEA, zmax=45.0, density=1.0, scale=None, towers=(), paint_kw=None):
    """Sea of clouds seen from above (perspective), heads on a plane added in painter's order.

    horizon: horizon row (fraction of H); cam_h: camera height above the mean cloud tops (world units);
    sun_px: sun position (px) - its column sets the sun azimuth; sun_elev: sun elevation (deg);
    splits: depths Z at which to cut the sea into layers (e.g. tower depths) so towers can be
            composited in between;  towers: [(X, Z, radius, height)] added to the light heightfield
            (they cast long shadows toward the camera).
    Returns dict(layers=[RGBA far..near], zrow=(H,) plane depth per row (for parallax), f=focal px,
                 project=function(X, Y, Z)->(x, y))."""
    sc = H / 1080.0 if scale is None else scale
    rng = np.random.default_rng(seed)
    f = (W / 2) / math.tan(math.radians(fov / 2))
    yh = horizon * H
    Yc = cam_h

    def proj(X, Y, Z):
        return W / 2 + f * X / Z, yh + f * (Yc - Y) / Z
    # sun direction (world): azimuth from the sun column, low elevation, ahead of the camera
    sx = (sun_px[0] - W / 2) / f if sun_px is not None else 0.3
    el = math.radians(sun_elev)
    Ld = np.array([sx, math.tan(el), 1.0])
    Ld /= np.linalg.norm(Ld)
    # --- heads (world): rows from far to near, size growing slowly with distance
    heads = []
    Z = zmax
    while Z > 1.0:
        r0 = 0.28 * Z ** 0.45
        dzr = 0.42 * r0 / density ** 0.5
        xw = Z * W / (2 * f) * 1.25 + 2 * r0
        X = -xw + rng.uniform(0, r0)
        while X < xw:
            r = r0 * rng.uniform(0.6, 1.5)
            flat = np.clip((Z - 8) / 40, 0, 1)
            h = r * rng.uniform(0.45, 0.85) * (1 - 0.65 * flat) * (1 + 0.6 * math.exp(-Z / 6.0))
            Xh, Zh = X + rng.uniform(-0.3, 0.3) * r, Z + rng.uniform(-0.5, 0.5) * dzr
            # clustering: big billowing mounds and sunken valleys (lower, smaller heads)
            mnd = _fbm3(Xh * 0.16 + 3.1, Zh * 0.2 + 1.7, 0.5, seed + 77, 3)
            mnd = min(max((mnd - 0.3) / 0.4, 0.0), 1.0)
            r_ = r * (0.7 + 0.7 * mnd)
            h_ = h * (0.45 + 1.25 * mnd)
            base = -0.55 * r0 * (1 - mnd) ** 1.5 * (1 - 0.7 * flat)
            heads.append((Xh, Zh, r_, h_, base))
            X += r * rng.uniform(1.1, 1.8) / density ** 0.5
        Z -= dzr
    heads.sort(key=lambda q: -q[1])                                    # far -> near
    # --- light heightfield (world XZ grid)
    gd = 0.08
    gx0, gz0 = -zmax * W / (2 * f) * 1.3 - 5, 0.0
    nx = int(-2 * gx0 / gd) + 1
    nzg = int((zmax + 8) / gd) + 1
    HF = np.full((nzg, nx), -2.0, np.float32)
    for (X, Z, r, h, base) in list(heads) + [(t[0], t[1], t[2], t[3], 0.0) for t in towers]:
        i0, i1 = int((X - r - gx0) / gd), int((X + r - gx0) / gd) + 2
        k0, k1 = int((Z - r - gz0) / gd), int((Z + r - gz0) / gd) + 2
        i0, k0 = max(i0, 0), max(k0, 0)
        i1, k1 = min(i1, nx), min(k1, nzg)
        if i1 <= i0 or k1 <= k0:
            continue
        kk, ii = np.mgrid[k0:k1, i0:i1].astype(np.float32)
        dx = (gx0 + ii * gd - X) / r
        dz = (gz0 + kk * gd - Z) / r
        dome = base + h * np.sqrt(np.clip(1 - dx * dx - dz * dz, 0, 1)) ** 0.8
        np.maximum(HF[k0:k1, i0:i1], dome, out=HF[k0:k1, i0:i1])
    HF = cv2.GaussianBlur(HF, (0, 0), 1.0)
    # --- rasterise heads in painter's order (far -> near) into one relief; remember the layer group
    cuts = sorted(splits, reverse=True)
    groups = []
    cur = []
    ci = 0
    for hd in heads:
        while ci < len(cuts) and hd[1] < cuts[ci]:
            groups.append(cur)
            cur = []
            ci += 1
        cur.append(hd)
    groups.append(cur)
    while len(groups) < len(cuts) + 1:
        groups.append([])
    Rf = Relief(W, H, seed, sc, cells=(70, 30, 12), sun_dir=(0.0, 1.0, 0.0))
    PZ = np.zeros((H, W), np.float32)
    LID = np.full((H, W), -1, np.int32)
    xs_ = np.arange(W, dtype=np.float32)[None, :]
    ys_ = np.arange(H, dtype=np.float32)[:, None]
    for gi_, grp in enumerate(groups):
        for (X, Z, r, h, base) in grp:
            if Z < 0.6:
                continue
            cx, ytop = proj(X, base + h, Z)
            a = f * r / Z
            if cx + a * 1.4 < 0 or cx - a * 1.4 > W or a < 1.2:
                continue
            th = math.atan2(Yc, Z)
            b = (h * math.cos(th) * 0.95 + r * math.sin(th) * 0.9) * f / Z * 0.55
            b = max(b, 0.8)
            cy = ytop + b
            if cy - b > H + 5:
                continue
            rr = min(a, b)
            bump = float(np.clip(rr / (45 * sc), 0.12, 1.0))
            if sun_px is not None:
                vx, vy = sun_px[0] - cx, -(sun_px[1] - cy)
                nn = math.hypot(vx, vy) + 1e-6
                Rf.sun2 = (0.55 * vx / nn, 0.55 * vy / nn + 0.6)
            Rf.mass(cx, cy, a, b, bb=b * 2.2, z0=-Z, zd=0.3 * rr, bump=bump, up_only=0.9, warp=0.06, zg=0.0,
                    tagZ=PZ, tagZval=Z, tagID=LID, tagval=gi_)
    M = (Rf.A > 0.5).astype(np.uint8)
    PX = (xs_ - W / 2) * PZ / f
    PY = Yc - (ys_ - yh) * PZ / f
    L = _hf_light(HF, gx0, gz0, gd, PX.astype(np.float32), PY.astype(np.float32), PZ, M, Ld[0], Ld[1], Ld[2],
                  60, 0.03, 1.08, 0.9, np.array([0.01, 0.05, 0.12]))
    # --- paint (shared painter), then aerial perspective by depth
    kw = dict(bands=(0.16,), head_t=0.42, head_t2=0.72, amb_z=0.0, shade_rim=0.5, head_rim=0.7, hdr=1.1, tear=0.0, rim=1.2, glow=0.0, backlit=0.0,
              smooth_px=8.0, scallop_px=10.0, sil_detail=0.5, shade_light=-0.25, abs_bands=True)
    if paint_kw:
        kw.update(paint_kw)
    sd2 = (Ld[0], 0.7, 0.0)
    P = _pal(pal)
    if sun_px is not None:
        az = np.exp(-np.abs(xs_ - sun_px[0]) / (0.35 * W)) * np.ones((H, 1), np.float32)
        pink_lit, pink_hi = _hex('#f4aab0'), _hex('#f8dcd8')
        kw.setdefault('pal_maps', dict(lit=_lerp(pink_lit, P['lit'], az), hi=_lerp(pink_hi, P['hi'], az)))
    rgba = paint_relief(Rf, L, pal, sd2, sun_px, H, 0, seed=seed, **kw)
    zr = np.where(PZ > 0, PZ, 60.0)
    fog = 1 - np.exp(-zr / (zmax * 0.45))
    fog = fblur(fog.astype(np.float32), 2.0)
    if sun_px is not None:
        dsun = np.sqrt((xs_ - sun_px[0]) ** 2 * 0.35 + (ys_ - sun_px[1]) ** 2) / W
        warm = np.exp(-dsun / 0.25)
    else:
        warm = np.zeros((H, W), np.float32)
    hz = _lerp(P['haze'], _hex('#ffcf8a'), np.clip(warm, 0, 1))
    rgba[..., :3] = _lerp(rgba[..., :3], hz, np.clip(fog * 0.8, 0, 1))
    layers = []
    for gi_ in range(len(groups)):
        lay = rgba.copy()
        m = (LID == gi_).astype(np.float32)
        mm = cv2.dilate(m, np.ones((3, 3), np.uint8))
        lay[..., 3] = rgba[..., 3] * np.where(LID >= 0, m, mm)
        layers.append(lay)
    rows = np.arange(H, dtype=np.float32)
    zrow = np.where(rows > yh + 1, f * Yc / np.maximum(rows - yh, 1e-3), 1e4).astype(np.float32)
    return dict(layers=layers, zrow=zrow, f=f, project=proj, rgba=rgba, L=L, sun_world=Ld, yh=yh, PZ=PZ)
