"""Mass painter: Shinkai-style cumulus built the way a background artist paints them.

A cloud = a few dozen MASSES painted back-to-front. Each mass is a union of billow discs (so its
silhouette is cauliflower-lobed), filled with a smooth painted gradient derived from
  * the macro envelope normal of the whole cloud at the mass (broad lit / shadow planes),
  * a gentle local "pillow" normal from the mass's own distance field (roundness near its edge),
quantised into soft painted value bands (lit / mid / shadow / core).
Where a mass overlaps the ones behind it, it casts a soft cool TUCK shadow on the side away from the
sun (that is what reads as the crisp crevice between lobes), and gets a thin warm LIP along its
sun-facing edge. Detail therefore lives on edges; interiors stay broad and painterly.

Main entry points:
  tower_masses(...)  / heap_masses(...)  -> list of mass dicts
  paint_masses(W, H, masses, sun_xy, pal) -> RGBA float32 plate
"""
import math
import numpy as np
import cv2

from . import core as C

PALETTES = {
    'noon': dict(lit=(1.00, 0.99, 0.96), mid=(0.84, 0.87, 0.96), shadow=(0.60, 0.66, 0.87),
                 core=(0.47, 0.53, 0.80), bounce=(0.66, 0.80, 0.97), lip=(1.0, 1.0, 1.0),
                 tuck=(0.52, 0.58, 0.84), haze=(0.74, 0.87, 0.98)),
    'sunset': dict(lit=(1.00, 0.82, 0.55), mid=(0.95, 0.60, 0.55), shadow=(0.55, 0.42, 0.66),
                   core=(0.36, 0.30, 0.56), bounce=(0.78, 0.50, 0.62), lip=(1.0, 0.95, 0.75),
                   tuck=(0.42, 0.32, 0.58), haze=(0.98, 0.72, 0.62)),
    'sunrise': dict(lit=(1.00, 0.88, 0.66), mid=(0.96, 0.70, 0.66), shadow=(0.50, 0.46, 0.76),
                    core=(0.30, 0.29, 0.62), bounce=(0.72, 0.58, 0.80), lip=(1.0, 0.97, 0.84),
                    tuck=(0.36, 0.34, 0.66), haze=(1.0, 0.84, 0.70)),
}


def _band_colour(ndl, P):
    lit, mid, sh, core = (np.array(P[k], np.float32) for k in ('lit', 'mid', 'shadow', 'core'))
    tc = C.smoothstep(-0.45, -0.10, ndl)[..., None]
    t1 = C.smoothstep(0.05, 0.20, ndl)[..., None]
    t2 = C.smoothstep(0.38, 0.58, ndl)[..., None]
    c = core + (sh - core) * tc
    c = c + (mid - c) * t1
    c = c + (lit - c) * t2
    return c


def _mass_mask(shape, discs, ss):
    """Union of discs -> AA mask in a local canvas. discs: (n,3) x,y,r in local ss pixels."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    for x, y, r in discs:
        cv2.circle(m, (int(round(x * 4)), int(round(y * 4))), int(round(r * 4)), 255, -1, cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255.0


def paint_masses(W, H, masses, sun_xy, pal='noon', ss=2, lip_px=2.2, lip_amt=0.9, tuck_amt=0.35,
                 pillow=1.1, ground_y=None, haze_amt=0.0, wisp_seed=0):
    """Paint masses (list of dicts: discs (n,3) px, z, n (3,) macro normal) into an RGBA plate."""
    P = PALETTES[pal] if isinstance(pal, str) else pal
    Ws, Hs = W * ss, H * ss
    rgb = np.zeros((Hs, Ws, 3), np.float32)
    alpha = np.zeros((Hs, Ws), np.float32)
    tuck_col = np.array(P['tuck'], np.float32)
    lip_col = np.array(P['lip'], np.float32)
    bounce = np.array(P['bounce'], np.float32)
    sx, sy = sun_xy[0] * ss, sun_xy[1] * ss
    allys = np.concatenate([m['discs'][:, 1] for m in masses])
    ytop, ybot = allys.min() * ss, allys.max() * ss
    for m in sorted(masses, key=lambda m: m['z']):
        d = m['discs'].astype(np.float32) * np.array([ss, ss, ss], np.float32)
        pad = int(d[:, 2].max() * 0.6) + 8
        x0 = int(max(0, (d[:, 0] - d[:, 2]).min() - pad)); x1 = int(min(Ws, (d[:, 0] + d[:, 2]).max() + pad))
        y0 = int(max(0, (d[:, 1] - d[:, 2]).min() - pad)); y1 = int(min(Hs, (d[:, 1] + d[:, 2]).max() + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        loc = d.copy(); loc[:, 0] -= x0; loc[:, 1] -= y0
        mask = _mass_mask((y1 - y0), (x1 - x0), loc, ss) if False else None
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for x, y, r in loc:
            cv2.circle(mask, (int(round(x * 4)), int(round(y * 4))), int(round(r * 4)), 255, -1, cv2.LINE_AA, shift=2)
        mask = mask.astype(np.float32) / 255.0
        hh, ww = mask.shape
        R0 = float(d[:, 2].max())
        nsx = C.value_noise(ww, hh, max(2, ww / (R0 * 0.5)), max(2, hh / (R0 * 0.5)), seed=int(m['z'] * 7) % 9973) - 0.5
        nsy = C.value_noise(ww, hh, max(2, ww / (R0 * 0.5)), max(2, hh / (R0 * 0.5)), seed=int(m['z'] * 7) % 9973 + 1) - 0.5
        mask = C.warp(mask, nsx * R0 * 0.12, nsy * R0 * 0.12, border=cv2.BORDER_CONSTANT)
        if 'clip_y' in m:  # flat-ish base
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            mask *= C.smoothstep(m['clip_y'] * ss + 6 * ss, m['clip_y'] * ss - 10 * ss, yy)
        # local pillow normal from distance field
        dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        R = d[:, 2].mean()
        hgt = np.sqrt(np.clip(dist / (R * 0.9), 0, 1))
        hgt = C.blur(hgt, R * 0.25)
        gy, gx = np.gradient(hgt)
        k = R * 0.9
        pnx, pny = -gx * k, -gy * k
        # macro normal + pillow
        mn = np.asarray(m['n'], np.float32)
        nx = mn[0] + pillow * pnx; ny = mn[1] + pillow * pny; nz = mn[2] + 0 * pnx
        nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        cx, cy = d[:, 0].mean(), d[:, 1].mean()
        lx, ly = sx - cx, sy - cy
        ln = math.hypot(lx, ly) + 1e-6
        L = np.array([lx / ln * 0.85, ly / ln * 0.85, 0.25], np.float32); L /= np.linalg.norm(L)
        ndl = (nx * L[0] + ny * L[1] + nz * L[2]) / nl
        col = _band_colour(ndl, P)
        # sky bounce toward lower part of the cloud, in shadow
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        low = np.clip((yy - ytop) / (ybot - ytop + 1e-6), 0, 1)
        col = col * (1.0 - 0.10 * low[..., None] * np.array([1.0, 0.7, 0.2], np.float32))
        col = col + (bounce - col) * (0.30 * low * (1 - C.smoothstep(0.0, 0.3, ndl)))[..., None]
        # sun-facing lip: thin band inside the silhouette where the edge faces the sun
        L2 = np.array([lx / ln, ly / ln], np.float32)
        er = cv2.erode((mask > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(lip_px * ss)))
        band = np.clip(mask - er.astype(np.float32), 0, 1)
        band = C.blur(band, 0.6 * ss)
        facing = np.clip(-(gx * L2[0] + gy * L2[1]) / (np.sqrt(gx * gx + gy * gy) + 1e-5), 0, 1)
        lipk = (band * facing ** 1.2 * lip_amt * (0.3 + 0.7 * C.smoothstep(-0.1, 0.5, ndl)))[..., None]
        col = col + (lip_col - col) * lipk
        # crease occlusion: the back mass darkens just ABOVE this mass's upper contour (its shaded
        # underside), plus a little on the side away from the sun
        mb = C.blur(mask, R * 0.08)
        gyy, gxx = np.gradient(mb)
        gm = np.sqrt(gxx * gxx + gyy * gyy) + 1e-5
        up_off = np.float32([[1, 0, -L2[0] * R * 0.12], [0, 1, -R * 0.30]])
        sh = cv2.warpAffine(mask, up_off, (x1 - x0, y1 - y0))
        sh = C.blur(sh, R * 0.22) * (1 - mask)
        a_behind = alpha[y0:y1, x0:x1]
        sub = rgb[y0:y1, x0:x1]
        tk = (sh * tuck_amt * a_behind)[..., None]
        sub += (tuck_col * a_behind[..., None] - sub) * tk
        # lost edges: the downward-facing contour of a mass dissolves softly into what is behind
        down = np.clip(-gyy / gm, 0, 1) ** 1.5 * C.smoothstep(0.0, 0.02, gm * R)
        soft = C.blur(mask, R * 0.22) * mask
        mask = mask * (1 - down) + np.minimum(mask, soft * 1.3) * down
        # composite the mass (premultiplied accumulation)
        mk = mask[..., None]
        sub += (col - sub) * mk
        a_behind += (1 - a_behind) * mask
    out = np.dstack([rgb, alpha])
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA)
    a = np.clip(out[..., 3:4], 1e-4, 1)
    out[..., :3] = np.clip(out[..., :3] / a, 0, 1.5)
    if haze_amt and ground_y is not None:
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        hz = C.smoothstep(ytop / ss, ground_y, yy) * haze_amt
        out[..., :3] = out[..., :3] + (np.array(P['haze'], np.float32) - out[..., :3]) * hz
    return out


def _lobed_discs(rng, cx, cy, r, n_small=6, spread=1.0, bias_up=0.6, levels=3):
    """A mass: an elongated core of discs + FRACTAL cauliflower bumps growing outward/upward
    (children r*0.3-0.55 on the parent's outer arc, `levels` deep)."""
    discs = []
    # core: a few overlapping discs along a slightly squashed horizontal ellipse
    ncore = 3
    for i in range(ncore):
        u = (i - (ncore - 1) / 2) / max(1, ncore - 1)
        discs.append((cx + u * r * 0.9, cy + abs(u) * r * 0.25 + rng.uniform(-0.1, 0.1) * r, r * (0.8 - 0.25 * abs(u))))
    frontier = list(discs)
    for lev in range(levels):
        nxt = []
        for (px, py, pr) in frontier:
            k = int(rng.integers(2, 5)) if lev else n_small
            for j in range(k):
                # outward direction from mass centre, biased upward
                ox, oy = px - cx, py - cy
                base = math.atan2(oy - bias_up * r * 1.5, ox + 1e-6)
                a = base + rng.uniform(-1.3, 1.3)
                if math.sin(a) > 0.55 and rng.random() < 0.8:   # few bumps on the underside
                    continue
                rr = pr * rng.uniform(0.28, 0.55)
                dd = pr * rng.uniform(0.75, 0.98)
                c = (px + math.cos(a) * dd, py + math.sin(a) * dd, rr)
                discs.append(c); nxt.append(c)
        frontier = nxt
    return np.array(discs, np.float32)


def tower_masses(W, H, cx, base_y, top_y, width, seed=0, lean=0.0, n_levels=6, per_level=6, crown=1.0):
    """Masses for a towering cumulus / cumulonimbus. Returns list of mass dicts."""
    rng = np.random.default_rng(seed)
    height = base_y - top_y
    knots = np.array([0.0, 0.15, 0.4, 0.65, 0.85, 1.0])
    prof = np.array([1.0, 0.95, 0.74, 0.62, 0.5 * crown, 0.28 * crown]) * (1 + rng.uniform(-0.07, 0.07, 6))
    masses = []
    for li in range(n_levels):
        y01 = (li + 0.5) / n_levels
        Ry = np.interp(y01, knots, prof) * width * 0.5
        yc = base_y - y01 * height * 0.92
        ax = cx + lean * y01 * height
        n = max(2, int(per_level * (0.6 + 0.6 * Ry / (width * 0.5))))
        for k in range(n):
            th = rng.uniform(-1.2, 1.2) if k else rng.uniform(-0.3, 0.3)
            r = Ry * rng.uniform(0.36, 0.55) * (1 - 0.3 * abs(math.sin(th)))
            x = ax + Ry * math.sin(th) * 0.8
            y = yc + rng.uniform(-0.5, 0.5) * height / n_levels
            up = 0.25 + 0.9 * C.smoothstep(0.7, 1.0, y01)
            nvec = np.array([math.sin(th), -up, math.cos(th)], np.float32); nvec /= np.linalg.norm(nvec)
            discs = _lobed_discs(rng, x, y, r, n_small=int(rng.integers(5, 9)))
            masses.append(dict(discs=discs, z=math.cos(th) * Ry + y01 * height * 0.35 + rng.uniform(0, 5), n=nvec, clip_y=base_y))
    # crown / dome masses
    Rc = np.interp(0.95, knots, prof) * width * 0.5
    for k in range(6):
        th = rng.uniform(-1.0, 1.0)
        r = Rc * rng.uniform(0.45, 0.7)
        x = cx + lean * height + Rc * math.sin(th) * 0.9
        y = top_y + r * 0.9 + rng.uniform(0, 0.4) * r
        nvec = np.array([math.sin(th) * 0.7, -0.9, math.cos(th) * 0.6], np.float32); nvec /= np.linalg.norm(nvec)
        masses.append(dict(discs=_lobed_discs(rng, x, y, r, 8, bias_up=0.8), z=height + k, n=nvec, clip_y=base_y))
    return masses


def heap_masses(W, H, cx, base_y, width, height, seed=0, n=10):
    """A small/medium fair-weather cumulus heap (flat base, domed top)."""
    rng = np.random.default_rng(seed)
    masses = []
    for k in range(n):
        u = rng.uniform(-1, 1)
        dome = math.sqrt(max(0.0, 1 - u * u))
        r = height * rng.uniform(0.28, 0.45) * (0.6 + 0.4 * dome)
        x = cx + u * width * 0.42
        y = base_y - r * 0.6 - dome * height * rng.uniform(0.2, 0.6)
        nvec = np.array([u * 0.8, -0.6 * dome - 0.1, 0.5 + 0.3 * dome], np.float32); nvec /= np.linalg.norm(nvec)
        masses.append(dict(discs=_lobed_discs(rng, x, y, r, 6), z=dome * 10 + rng.uniform(0, 2), n=nvec, clip_y=base_y))
    return masses


def tower_masses2(W, H, cx, base_y, top_y, width, seed=0, lean=0.0, scales=((0.26, 10), (0.15, 22), (0.08, 40)),
                  crown=1.0):
    """Hierarchical tower: big masses first, then medium and small bulges painted over them (nearer
    the viewer), all scattered on the front surface of a tower envelope. Returns mass dicts."""
    rng = np.random.default_rng(seed)
    height = base_y - top_y
    knots = np.array([0.0, 0.12, 0.35, 0.6, 0.82, 1.0])
    prof = np.array([1.0, 0.85, 0.62, 0.55, 0.50 * crown, 0.25 * crown]) * (1 + rng.uniform(-0.06, 0.06, 6))
    masses = []
    for si, (sc, count) in enumerate(scales):
        for k in range(count):
            y01 = rng.uniform(0.03, 0.97) ** (0.85 if si == 0 else 1.0)
            Ry = np.interp(y01, knots, prof) * width * 0.5
            th = rng.uniform(-1.25, 1.25)
            r = width * sc * rng.uniform(0.75, 1.2) * (0.55 + 0.45 * Ry / (width * 0.5))
            x = cx + lean * y01 * height + Ry * math.sin(th) * 0.85
            y = base_y - y01 * height * 0.93
            up = 0.2 + 0.9 * C.smoothstep(0.7, 1.0, y01)
            nvec = np.array([math.sin(th), -up, math.cos(th)], np.float32); nvec /= np.linalg.norm(nvec)
            discs = _lobed_discs(rng, x, y, r, n_small=int(rng.integers(4, 8)), levels=3 - (si == 2))
            z = si * 1000 + math.cos(th) * Ry + rng.uniform(0, 20)
            masses.append(dict(discs=discs, z=z, n=nvec, clip_y=base_y))
    return masses
