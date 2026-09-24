"""clouds3_Y - painted Shinkai-style clouds from 3D puff hierarchies ("puff painter").

How a Shinkai background painter builds a cumulus (from out/refs: wwy_01, gow_02, wwy_03, cm5_04,
cm5_05, yn_02, yn_03, wwy_02) and how this module reproduces each decision
--------------------------------------------------------------------------------------------------
1. SHAPE HIERARCHY.  A cloud is a few big masses, each carrying heads, each head carrying florets
   (3-4 scales, cauliflower).  -> Clouds are built as hierarchies of 3D spheres ("puffs"): primaries
   laid along an envelope (tower columns / dome / sea surface), children spawned on the upper / front
   hemisphere of their parent at 0.35-0.55 of its size, down to florets of a few px.  Bases are flat
   (a per-puff floor with a soft fade), anvils are flat-topped (a per-puff ceiling).
2. VALUE MASSES, NOT SPHERES.  Painters light the BIG form (one terminator per tower), then paint a
   LIGHT CAP on the sunward edge of each head - a flat bright shape whose outer edge is the crisp
   cauliflower contour and whose inner edge fades into a flat cool body (gow_02: the interior heads
   read ONLY through these caps). -> form = normals of a silhouette "pillow"; caps = a head-aware
   march toward the sun through the z-buffer: the distance until the ray leaves its own head (into
   sky, or onto a head lying behind) is the cap; a head in front buries it. Lit shapes bulge into the
   florets straddling their edge (scalloped light shapes, wwy_01). A reverse march paints the crease:
   the dark underside of a rear head right behind a front head's lit top. No per-puff sphere shading.
3. FEW VALUE STEPS.  3-5 painted values (deep / shade / mid / lit_lo / lit / hi): soft posterisation
   with brush-modulated edge hardness + a Kuwahara "painter pass" that groups values into flat brushy
   patches, strongest in shadow (detail is lost in the dark, kept in the light).
4. HARD vs LOST EDGES.  Crisp outline where lit; shadow-side silhouette feathers into the air colour;
   flat bases fade softly (per-puff floor); anvils are flat-topped (per-puff ceiling).
5. LIGHT BLEEDS OVER THE SILHOUETTE.  HDR silver/gold lining just inside the sunlit outline (scene
   bloom + halo() spill it into the sky). BACKLIT (sun_z < 0: sea of clouds at sunrise, yn_02 /
   cm5_05): bodies go dark, caps shrink to thin hot crest lines on every sunward top edge that drops
   onto something well behind it; hotter on the sun axis.
6. SATURATION SITS AT THE TERMINATOR and in the cool shadow (blue-violet), never in the highlights.
   Aerial perspective pulls distant puffs toward the air colour (per-puff haze); sea heaps are
   flattened ellipsoids that compress into thin scalloped rows toward the horizon; low-lying heaps
   (valleys) are pushed toward deep indigo.

Everything is procedural (numpy / cv2 / numba); deterministic for a given seed.

API  (all sizes in plate px; scale with W; colours display RGB, rim may exceed 1 = HDR)
===
Palettes
    PALETTES: 'noon', 'magic_hour', 'sunset', 'sunrise'   keys: deep shade mid lit_lo lit hi rim air
              sky (up-facing shadow tint) bounce (light from below)
    palette(name_or_dict) -> dict of float32 arrays;  mix_palette(a, b, t) (time-of-day blends)
Sun: every plate takes sun=(x, y) in plate px (may be far off-plate = directional light) and sun_z in
    -1..1 = elevation of the light TOWARD the viewer (+0.1..0.4 front/side lit; <0 backlit).
Plates (one call -> RGBA float32 (h, w, 4), straight alpha, colour bled into transparent margin)
    cumulonimbus_plate(w, h, cx, base_y, height, sun, sun_z=0.35, pal='noon', seed=0, width=None,
                       anvil=0.0, anvil_dir=1, turrets=3, lean=0.0, haze=0.0, levels=3, **paint_kw)
    cumulus_plate(w, h, clouds=[(cx, base_y, cw, ch, seed[, haze]) ...], sun, sun_z, pal, **paint_kw)
                       fair-weather heaps (several per plate, back to front)
    horizon_bank_plate(w, h, y, height, sun, sun_z, pal, seed, rows=2, haze=0.45, flat=0.8,
                       spread=(1.6, 3.6), **paint_kw)   low bank / (flat=0.95, value_bias<0) dark shelf
    sea_of_clouds_plate(w, h, horizon_y, sun, sun_z=-0.3, pal='sunrise', seed=0, splits=(), cam_h=2.5,
                        relief=0.6, cell=0.9, valley=0.25, air=None, haze_far=0.85, far=110,
                        plate_kw=None, **paint_kw) -> [plates far -> near]; far plate opaque
    sea_tower_plate(w, h, horizon_y, x, depth, height, sun, sun_z=0, pal, seed, cam_h=2.5, width=0.6,
                    turrets=3, anvil=0, lean=0, haze=0, **paint_kw)   tower standing in the sea at world
                    depth D (composite it after the sea plate that ends at D: splits=(.., D, ..))
    cirrus_plate(w, h, sun, pal='noon', seed=0, region=(0.02, 0.45), angle=-12, density=0.5,
                 mackerel=0.5, opacity=0.8, color=None, fleck=None, streaks=1.0)
                 mackerel dry-brush flecks in rippled rows + tapered cirrus strokes
    stratus_band(w, h, y, thick, sun, pal, seed, color=None, lit_color=None, opacity=0.9, layers=3)
                 flat stroke-built stratus bars (dusk bars, thin shelves)
Geometry (puff sets: dict of arrays x y z r floor fade ceil haze val sq h0 h1) + renderer
    tower_puffs(...), heap_puffs(...), bank_puffs(...), sea_puffs(...), sea_tower_puffs(...),
    sea_view(w, h, horizon_y, cam_h), merge(*puffs)
    paint(puffs, w, h, sun, sun_z, pal, **kw)   (see _paint docstring for every painting control)
    kuwahara(img, r, aniso=1, angle=0)          the painter-pass filter (usable on any image)
Drift / compositing
    drift(plate, W, H, t, speed=(0.003, 0), cam=(0, 0), zoom=1, depth=1, billow=0.0, seed=0)
        wind drift (fraction of W / s) + camera parallax (cam * depth) + zoom, optional slow coherent
        billowing (0.0008-0.0015 = calm churn). Deterministic, sub-pixel, flicker-free.
    screen_pos(p, W, H, plate_shape, cam, zoom, depth, t, speed) plate px -> screen px (sun, anchors)
    over(img, *layers)             straight-alpha composite of sampled RGBA layers (numba)
    halo(layer, strength, sigma)   additive halation from HDR linings (before bloom)
Cost at 1080p (plates 1.16x frame): tower ~8 s, heaps ~2 s (painted inside their bounding box), bank
~7 s, cirrus ~1 s, 3-plate sea ~17 s. Frames only sample plates: demo scenes ~1.0-1.5 s/frame.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C
from . import sky as _S
from . import fx as _F

# ============================================================================== palettes


def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, np.float32)


PALETTES = {
    # deep -> shade -> mid -> lit_lo (terminator, most saturated warm) -> lit -> hi ; rim = lining (HDR)
    'noon': dict(deep='#4a76ba', shade='#7299d2', mid='#9dbde4', lit_lo='#d9e0e6', lit='#f4eedc',
                 hi='#fffaea', rim=(1.18, 1.16, 1.08), air='#9ec4e8', sky='#9fc2ee', bounce='#8fb0dc'),
    'magic_hour': dict(deep='#56589a', shade='#8374ae', mid='#c28eb0', lit_lo='#f5a597', lit='#ffd1a4',
                       hi='#fff0d4', rim=(1.25, 1.1, 0.9), air='#d6a6bc', sky='#9b98d4', bounce='#e0a0b0'),
    'sunset': dict(deep='#3a2e6c', shade='#654a8a', mid='#ad5a8a', lit_lo='#f0766a', lit='#ffa86a',
                   hi='#ffd89c', rim=(1.3, 1.02, 0.7), air='#bd7a9a', sky='#7a68b0', bounce='#c86a88'),
    'sunrise': dict(deep='#262a58', shade='#454c80', mid='#85789c', lit_lo='#ec9c84', lit='#ffc47e',
                    hi='#ffe8b4', rim=(1.35, 1.1, 0.75), air='#d6a49a', sky='#6f78b0', bounce='#b88a9a'),
}
_ORDER = ('deep', 'shade', 'mid', 'lit_lo', 'lit', 'hi')
_STOPS = np.array([0.0, 0.2, 0.4, 0.58, 0.78, 1.0], np.float32)


def palette(p):
    if isinstance(p, str):
        p = PALETTES[p]
    return {k: _c(v).astype(np.float32) for k, v in p.items()}


def mix_palette(a, b, t):
    a, b = palette(a), palette(b)
    return {k: (a[k] * (1 - t) + b[k] * t).astype(np.float32) for k in a}


# ============================================================================== geometry: puffs

_KEYS = ('x', 'y', 'z', 'r', 'floor', 'fade', 'ceil', 'haze', 'val', 'sq', 'h0', 'h1')


def _empty():
    return {k: np.zeros(0, np.float32) for k in _KEYS}


def merge(*ps):
    """Concatenate puff sets (head indices are re-based)."""
    out = _empty()
    for p in ps:
        off = len(out['x'])
        for k in _KEYS:
            v = p[k].astype(np.float32)
            if k in ('h0', 'h1'):
                v = v + off
            out[k] = np.concatenate([out[k], v])
    return out


def _pack(lst):
    if not lst:
        return _empty()
    a = np.array(lst, np.float64)
    return {k: a[:, i].astype(np.float32) for i, k in enumerate(_KEYS)}


def _grow(prims, rng, levels, min_r, up=1.0, front=0.2, kids=(5, 8), scale=(0.28, 0.6), down_ok=0.25,
          side_bias=0.0):
    """Recursive cauliflower growth. prims: list of [x, y, z, r, floor, fade, ceil, haze].
    Children sit on the parent's upper/front hemisphere, poking out by ~0.6-0.9 of their radius.
    Every puff records its ancestors: h0 = primary head, h1 = first-generation sub-head (painters
    shade per head; florets only texture the outline)."""
    out = []
    for i, p in enumerate(prims):
        p = list(p) + [0.0] * (9 - len(p))
        p = p[:9] + [p[9] if len(p) > 9 else 1.0]
        out.append(p[:10] + [i, i])
    cur = list(range(len(out)))
    for lev in range(levels):
        nxt = []
        for pi in cur:
            p = out[pi]
            x, y, z, r = p[0], p[1], p[2], p[3]
            if r * scale[1] < min_r:
                continue
            n = rng.integers(kids[0], kids[1] + 1)
            tries = 0
            made = 0
            while made < n and tries < n * 4:
                tries += 1
                v = rng.normal(0, 1, 3)
                v /= np.linalg.norm(v) + 1e-9
                v[1] -= up * 0.8
                v[2] += front
                v[0] += side_bias
                v /= np.linalg.norm(v) + 1e-9
                if v[1] > down_ok:          # no florets on the underside: flat, calm bases
                    continue
                cr = r * rng.uniform(*scale) * (1.15 if lev == 0 else 1.0)
                d = r - cr * rng.uniform(0.15, 0.45)
                idx = len(out)
                h1 = idx if lev == 0 else p[11]
                sq = p[9]
                out.append([x + v[0] * d, y + v[1] * d * sq, z + v[2] * d, cr, p[4], p[5], p[6], p[7], p[8], sq,
                            p[10], h1])
                nxt.append(idx)
                made += 1
        cur = nxt
    return out


def tower_puffs(cx, base_y, height, width, seed=0, lean=0.0, turrets=3, anvil=0.0, anvil_dir=1.0, levels=3,
                z0=0.0, haze=0.0, min_r=None, fade=None, side=0.0):
    """Towering cumulus / cumulonimbus: a broad heaped base, a main column and `turrets` secondary
    columns of stacked heads; `anvil` 0..1 adds a flat-topped anvil sheet (length ~anvil*width*1.6)."""
    rng = np.random.default_rng(seed)
    min_r = min_r or max(2.0, height * 0.006)
    fade = fade if fade is not None else height * 0.05
    top = base_y - height
    ceil = top - 1e6
    if anvil > 0:
        ceil = top + height * 0.02
    prims = []

    def P(x, y, r, z=None, fl=base_y, ce=ceil, hz=haze):
        zz = z0 + (rng.normal(0, 0.25) * r if z is None else z)
        prims.append([x, y, zz, r, fl + rng.normal(0, 0.01) * height, fade, ce, hz])

    # columns: (x offset, top height fraction, width fraction)
    cols = [(0.0, 1.0, 0.55)]
    for k in range(turrets):
        s = -1 if k % 2 == 0 else 1
        cols.append((s * rng.uniform(0.18, 0.42), rng.uniform(0.45, 0.8), rng.uniform(0.3, 0.42)))
    for ox, hf, wf in cols:
        hcol = height * hf
        r0 = width * wf * 0.5
        yb = base_y - r0 * 0.5
        y = yb
        u = 0.0
        while u < 1.0:
            u = (base_y - y) / hcol
            rr = r0 * (1.0 - (0.6 - 0.45 * min(anvil * 2, 1)) * u ** 1.2) * rng.uniform(0.75, 1.2)
            zig = 0.28 if int((base_y - y) / max(r0, 1)) % 2 else -0.28
            xl = cx + width * ox * (1 - 0.5 * u) + lean * height * u + (zig + rng.normal(0, 0.1)) * rr
            P(xl, y, rr)
            # flanking heads so the column is not a string of beads
            if rng.random() < 0.45:
                P(xl + rng.choice([-1, 1]) * rr * rng.uniform(0.5, 0.8), y + rr * rng.uniform(-0.1, 0.3),
                  rr * rng.uniform(0.55, 0.75))
            y -= rr * rng.uniform(0.75, 0.95)
    # broad heaped base
    nb = 5
    for i in range(nb):
        f = (i + 0.5) / nb * 2 - 1
        rr = width * rng.uniform(0.16, 0.24)
        P(cx + f * width * 0.62, base_y - rr * rng.uniform(0.0, 0.25), rr)
    if anvil > 0:
        # wedge-shaped anvil: thick over the column, tapering outward, underside fanning up to the tips
        L = width * (0.5 + 1.8 * anvil)
        th = height * 0.1
        n = int(16 + 30 * anvil)
        for i in range(n):
            f = i / (n - 1) * 1.35 - 0.35                  # mostly downwind (anvil_dir)
            af = abs(f)
            xa = cx + anvil_dir * f * L + lean * height
            rr = th * (1.15 - 0.7 * af) * rng.uniform(0.8, 1.15)
            ya = top + th * 0.7 + (1 - af) ** 2 * th * 1.3 + rng.normal(0, 0.1) * th
            P(xa, ya, rr, ce=ceil)
    out = _grow(prims, rng, levels, min_r, kids=(5, 8), side_bias=side)
    return _pack(out)


def heap_puffs(cx, base_y, width, height, seed=0, levels=3, z0=0.0, haze=0.0, flat=0.5, min_r=None, fade=None):
    """Fair-weather cumulus heap: dome of primary heads on a flat base."""
    rng = np.random.default_rng(seed)
    min_r = min_r or max(1.5, height * 0.02)
    fade = fade if fade is not None else height * 0.2
    prims = []
    n = max(3, int(width / max(height, 1) * 3.0))
    for i in range(n):
        f = (i + rng.uniform(0.2, 0.8)) / n * 2 - 1
        dome = math.sqrt(max(1 - f * f, 0.05))
        rr = height * (0.14 + 0.55 * dome ** 1.3) * rng.uniform(0.8, 1.15)
        y = base_y - height * dome * (1 - flat * 0.5) + rr * 0.9
        y = min(y, base_y - rr * 0.25)
        prims.append([cx + f * width * 0.45, y, z0 + rng.normal(0, 0.3) * rr, rr,
                      base_y + rng.normal(0, 0.02) * height, fade, -1e9, haze])
    return _pack(_grow(prims, rng, levels, min_r, kids=(4, 7)))


def bank_puffs(x0, x1, y, height, seed=0, rows=2, haze=0.35, z0=0.0, levels=2, flat=0.8, spread=(1.6, 3.6)):
    """Horizon cloud bank: rows of low heaps, back rows smaller and hazier."""
    rng = np.random.default_rng(seed)
    out = _empty()
    for r in range(rows):
        k = r / max(rows - 1, 1)
        hh = height * (1.0 - 0.45 * k)
        by = y - height * 0.25 * k
        x = x0 - hh
        while x < x1 + hh:
            wv = hh * rng.uniform(*spread)
            hv = hh * rng.uniform(0.45, 1.0)
            out = merge(out, heap_puffs(x + wv / 2, by, wv, hv, seed=int(rng.integers(1 << 30)), levels=levels,
                                        z0=z0 - r * hh * 3, haze=haze + (1 - haze) * 0.45 * k, flat=flat))
            x += wv * rng.uniform(0.55, 0.9)
    return out


def sea_view(w, h, horizon_y, cam_h=2.5, focal=None):
    """Camera for the sea of clouds: returns dict(f, hy, cam_h, d_bottom) and project(X, Y, D)."""
    f = focal or w * 0.9
    return dict(f=f, hy=horizon_y, cam_h=cam_h, d_bottom=cam_h * f / max(h - horizon_y, 1.0))


def sea_puffs(w, h, horizon_y, seed=0, cam_h=2.5, focal=None, d_range=(None, 60.0), cell=0.9, relief=0.6,
              levels=3, haze_far=0.85, min_px=1.0, x_span=1.3, valley=0.25, far_ref=60.0, row_gap=0.8, squash=0.6):
    """Sea of clouds seen from above: puff heaps on a world plane (camera cam_h world units above the
    mean cloud top, horizon at horizon_y, world cell ~ one heap), perspective-projected. Only heaps with
    depth in d_range=(near, far) are produced (split the sea into parallax plates / sandwich towers).
    Low-lying heaps (valleys) get a negative value bias (`valley`): deep indigo troughs.
    z = -focal*ln(D): locally px-consistent depth, so screen-space light marching works."""
    rng = np.random.default_rng(seed)
    V = sea_view(w, h, horizon_y, cam_h, focal)
    f = V['f']
    d0 = d_range[0] if d_range[0] is not None else V['d_bottom'] * 0.75
    d1 = d_range[1]
    prims = []
    ph = [rng.uniform(0, 6.28) for _ in range(6)]

    def Ht(X, D):
        v = (np.sin(X * 0.33 + ph[0]) * np.sin(D * 0.21 + ph[1]) * 0.55
             + np.sin(X * 0.9 - D * 0.55 + ph[2]) * 0.3 + np.sin(X * 2.1 + D * 1.7 + ph[3]) * 0.15)
        return v * relief
    D = max(V['d_bottom'] * 0.75, 0.5)
    rowi = 0
    while D < d1:
        cs = cell * (1 + 0.015 * D)
        rowi += 1
        if D >= d0:
            # one ridge (row): heaps overlapping tightly along X, height following the relief field,
            # occasionally broken (gaps show the valley behind)
            half = x_span * (w / 2) * D / f + cs * 3
            X = -half + rng.uniform(0, cs)
            gap_ph = rng.uniform(0, 6.28)
            while X < half:
                hd = D + rng.normal(0, 0.06) * cs
                top = Ht(X, hd)
                brk = math.sin(X * 0.45 / cs + gap_ph) + 0.4 * math.sin(X * 1.3 / cs + gap_ph * 2)
                if brk < -1.05 + 0.5 * relief * 0:
                    X += cs * 0.5
                    continue
                sz = 0.6 + 0.8 * (0.5 + 0.5 * math.sin(X * 0.23 + ph[4]) * math.sin(hd * 0.31 + ph[5]))
                R = cs * rng.uniform(0.4, 0.8) * sz * (1 + 0.35 * top / max(relief, 1e-3))
                Y = top - R * 0.7
                sx = w / 2 + X * f / hd
                sy = horizon_y + (cam_h - Y) * f / hd
                sr = R * f / hd
                if sr >= min_px * 0.5 and -sr < sx < w + sr and sy - sr < h:
                    z = -f * math.log(hd)
                    hz = haze_far * min(1.0, hd / far_ref) ** 1.1
                    vb = valley * min(top / max(relief, 1e-3), 0.3)
                    prims.append([sx, sy, z, sr, 1e9, 1.0, -1e9, hz, vb, squash * (0.55 + 0.45 * min(1.0, 8.0 / hd))])
                X += cs * rng.uniform(0.3, 0.55)
        D += cs * row_gap * max(1.0, D / 8.0)
    lv = levels if d0 > V['d_bottom'] * 1.5 else max(levels - 1, 1)
    return _pack(_grow(prims, rng, lv, max(min_px, 1.0), up=1.4, front=0.1, kids=(4, 7),
                       scale=(0.3, 0.55), down_ok=0.0))


def sea_tower_puffs(w, h, horizon_y, x, depth, height, seed=0, cam_h=2.5, focal=None, width=0.6, turrets=3,
                    anvil=0.0, lean=0.0, haze=0.0, levels=3):
    """A cumulus tower standing in the sea of clouds at screen x (fraction of w) and world depth D;
    height in world units (sea cell ~0.9). Its base sits below the sea surface (hidden by nearer rows)."""
    V = sea_view(w, h, horizon_y, cam_h, focal)
    f = V['f']
    sy = horizon_y + (cam_h + 1.4) * f / depth
    H = height * f / depth
    return tower_puffs(x * w, sy, H, width * H, seed=seed, lean=lean, turrets=turrets, anvil=anvil, levels=levels,
                       z0=-f * math.log(depth), haze=haze, fade=H * 0.02)


# ============================================================================== numba kernels

@njit(cache=True, fastmath=True)
def _splat(X, Y, Z, R, FL, FF, CE, SQ, h, w, zb, nx, ny, nz, cov, sid):
    for k in range(X.shape[0]):
        r = R[k]
        if r <= 0.3:
            continue
        sq = SQ[k]
        x0 = max(int(X[k] - r - 2), 0)
        x1 = min(int(X[k] + r + 3), w)
        y0 = max(int(Y[k] - r * sq - 2), 0)
        y1 = min(int(Y[k] + r * sq + 3), h)
        fl = FL[k]
        ff = max(FF[k], 0.5)
        ce = CE[k]
        rr1 = (r + 1.0) * (r + 1.0)
        for i in range(y0, y1):
            fy = 1.0
            if i > fl - ff:
                fy = (fl - i) / ff
                if fy <= 0.0:
                    continue
            if i < ce:
                fy2 = 1.0 - (ce - i)
                if fy2 <= 0.0:
                    continue
                fy = min(fy, fy2)
            dy = (i - Y[k]) / sq
            for j in range(x0, x1):
                dx = j - X[k]
                d2 = dx * dx + dy * dy
                if d2 > rr1:
                    continue
                d = math.sqrt(d2)
                c = min(max(r - d + 0.5, 0.0), 1.0) * fy
                if c <= 0.0:
                    continue
                if c > cov[i, j]:
                    cov[i, j] = c
                q = r * r - d2
                zz = Z[k] + math.sqrt(q if q > 0.0 else 0.0)
                if zz > zb[i, j]:
                    zb[i, j] = zz
                    a = dx / r
                    b = dy / r
                    s = a * a + b * b
                    if s > 1.0:
                        s2 = math.sqrt(s)
                        a /= s2
                        b /= s2
                        s = 1.0
                    nx[i, j] = a
                    ny[i, j] = b
                    nz[i, j] = math.sqrt(1.0 - s)
                    sid[i, j] = k


@njit(cache=True, fastmath=True, parallel=True)
def _caps(zb, cov, hid, HX, HY, HZ, HR, capw, sunx, suny, maxk, ck, mode, dzo):
    """Head-aware light march.
    mode 0 (caps): distance (px) from each pixel toward the sun until the ray leaves its head: into
      sky, or onto another head lying BEHIND / beyond this head's dome (-> lit cap). If another head
      IN FRONT covers the path first, the pixel is buried: distance + penalty (shade).
    mode 1 (crease): march AWAY from the sun; 1 - t/len if a head in front is met (the rear head's
      shaded underside right behind a front head's lit top edge), else 0."""
    h, w = zb.shape
    out = np.zeros((h, w), np.float32)
    for i in prange(h):
        for j in range(w):
            if cov[i, j] <= 0.01:
                continue
            dx = sunx - j
            dy = suny - i
            if mode == 1:
                dx = -dx
                dy = -dy
            dl = math.sqrt(dx * dx + dy * dy) + 1e-6
            dx /= dl
            dy /= dl
            cw = capw[i, j]
            hp = hid[i, j]
            ax, ay, az, ar = HX[hp], HY[hp], HZ[hp], HR[hp] * 1.3
            thr = min(0.5, cov[i, j] * 0.6)
            lim = cw * maxk
            res = lim + cw * 2.0 if mode == 0 else 0.0
            t = 1.0
            while t < lim:
                x = j + dx * t
                y = i + dy * t
                xi = int(x + 0.5)
                yi = int(y + 0.5)
                if xi < 0 or yi < 0 or xi >= w or yi >= h or cov[yi, xi] < thr:
                    if mode == 0:
                        res = t
                        dzo[i, j] = 1e4
                    break
                hq = hid[yi, xi]
                if hq != hp:
                    ddx = x - ax
                    ddy = y - ay
                    q = ar * ar - ddx * ddx - ddy * ddy
                    front = False
                    if q > 0.0:
                        za = az + math.sqrt(q)
                        if zb[yi, xi] > za:
                            front = True
                    if mode == 0:
                        res = t + (cw * ck if front else 0.0)
                        if not front:
                            dzo[i, j] = zb[i, j] - zb[yi, xi]
                        break
                    else:
                        if front:
                            res = 1.0 - t / lim
                        break
                t += max(1.0, t * 0.04)
            out[i, j] = res
    return out


@njit(cache=True, fastmath=True)
def _ellipses(X, Y, A, B, ANG, OP, LX, LY, LS, h, w, alpha, lit):
    """Splat soft-edged rotated ellipses (brush dabs). alpha = max coverage * opacity; lit = coverage of
    the same dab shrunk (LS) and offset toward the light by (LX, LY) px: the dab's lit side."""
    for k in range(X.shape[0]):
        a = A[k]
        b = B[k]
        r = max(a, b) + 2.0
        x0 = max(int(X[k] - r), 0)
        x1 = min(int(X[k] + r + 1), w)
        y0 = max(int(Y[k] - r), 0)
        y1 = min(int(Y[k] + r + 1), h)
        c = math.cos(ANG[k])
        s = math.sin(ANG[k])
        for i in range(y0, y1):
            for j in range(x0, x1):
                dx = j - X[k]
                dy = i - Y[k]
                u = (dx * c + dy * s) / a
                v = (-dx * s + dy * c) / b
                d = math.sqrt(u * u + v * v)
                cv_ = min(max((1.0 - d) * min(a, b) + 0.5, 0.0), 1.0) * OP[k]
                if cv_ > alpha[i, j]:
                    alpha[i, j] = cv_
                dx2 = dx - LX[k]
                dy2 = dy - LY[k]
                u = (dx2 * c + dy2 * s) / (a * LS[k])
                v = (-dx2 * s + dy2 * c) / (b * LS[k])
                d = math.sqrt(u * u + v * v)
                cl = min(max((1.0 - d) * min(a, b) * LS[k] + 0.5, 0.0), 1.0) * min(cv_ * 3.0, 1.0)
                if cl > lit[i, j]:
                    lit[i, j] = cl


# ============================================================================== helpers

def _blur(x, s):
    if s <= 0.3:
        return x
    return _F.fast_blur(x.astype(np.float32), s) if s > 6 else cv2.GaussianBlur(x, (0, 0), s)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(w, h, cells, seed, octaves=3, angle=0.0, stretch=1.0):
    """fbm 0..1, optionally anisotropic (brush direction `angle` deg, elongated by `stretch`)."""
    if stretch == 1.0 and angle == 0.0:
        return C.fbm(w, h, cells, octaves, seed=seed)
    big = int(max(w, h) * 1.5)
    n = C.fbm(big, big, cells * big / w, octaves, seed=seed, aspect=True)
    n = cv2.resize(n, (big, max(int(big / stretch), 4)), interpolation=cv2.INTER_AREA)
    n = cv2.resize(n, (big, big), interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((big / 2, big / 2), angle, 1.0)
    n = cv2.warpAffine(n, M, (big, big), borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (big - h) // 2, (big - w) // 2
    return np.ascontiguousarray(n[y0:y0 + h, x0:x0 + w])


def kuwahara(img, r, sectors=4, aniso=1.0, angle=0.0):
    """Fast (box-filter) Kuwahara filter: every pixel takes the mean colour of the least-varied of 4
    quadrant windows of radius r -> flat, brush-like value patches with preserved edges (the painted
    look). aniso > 1 stretches windows along `angle` (deg) for directional strokes."""
    r = max(int(r), 1)
    if aniso != 1.0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rw, rh = max(int(r * aniso), 1), r
    else:
        rot = img
        rw, rh = r, r
    lum = rot[..., :3].mean(-1)
    sx_, sy_ = rw * 0.5, rh * 0.5
    m = cv2.GaussianBlur(rot, (0, 0), sigmaX=sx_, sigmaY=sy_, borderType=cv2.BORDER_REFLECT)
    ml = cv2.GaussianBlur(lum, (0, 0), sigmaX=sx_, sigmaY=sy_, borderType=cv2.BORDER_REFLECT)
    m2 = cv2.GaussianBlur(lum * lum, (0, 0), sigmaX=sx_, sigmaY=sy_, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(m2 - ml * ml, 0)
    # 8 soft sectors: pick (softly) the calmest neighbourhood -> round flat patches, crisp edges kept
    acc = np.zeros_like(m)
    wsum = np.zeros(var.shape, np.float32)
    for k in range(8):
        a = k * math.pi / 4
        dx, dy = math.cos(a) * rw * 0.6, math.sin(a) * rh * 0.6
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        mm = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), borderMode=cv2.BORDER_REFLECT)
        vv = cv2.warpAffine(var, M, (m.shape[1], m.shape[0]), borderMode=cv2.BORDER_REFLECT)
        wt = 1.0 / (1e-5 + vv * 400.0) ** 4
        acc += mm * wt[..., None]
        wsum += wt
    best = acc / (wsum[..., None] + 1e-20)
    if aniso != 1.0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
        best = cv2.warpAffine(best, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return best.astype(np.float32)


def _ramp(v, pal):
    cols = np.stack([pal[k] for k in _ORDER])
    out = np.empty(v.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, _STOPS, cols[:, c])
    return out


# ============================================================================== renderer

def paint(puffs, w, h, sun, sun_z=0.35, pal='noon', floor_rgb=None, **kw):
    """Paint a puff set into a straight-alpha RGBA plate (h, w, 4). See _paint for all parameters.
    Clouds much smaller than the plate are painted inside their bounding box (+ margin) only."""
    if floor_rgb is None and len(puffs['x']):
        R = puffs['r']
        sq = np.where(puffs['sq'] > 0, puffs['sq'], 1.0)
        fs = kw.get('form_sigma') or float(np.percentile(R, 97)) * 0.9
        mg = int(fs * 1.5 + 16)
        x0 = int(max(np.min(puffs['x'] - R) - mg, 0))
        x1 = int(min(np.max(puffs['x'] + R) + mg, w))
        y0 = int(max(np.min(puffs['y'] - R * sq) - mg, 0))
        y1 = int(min(np.max(puffs['y'] + R * sq) + mg, h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return np.zeros((h, w, 4), np.float32)
        if (x1 - x0) * (y1 - y0) < 0.8 * w * h:
            P = dict(puffs)
            P['x'] = puffs['x'] - x0
            P['y'] = puffs['y'] - y0
            P['floor'] = puffs['floor'] - y0
            P['ceil'] = puffs['ceil'] - y0
            kw['form_sigma'] = fs
            kw.setdefault('unit', w / 1920.0)
            sub = _paint(P, x1 - x0, y1 - y0, (sun[0] - x0, sun[1] - y0), sun_z, pal, **kw)
            out = np.zeros((h, w, 4), np.float32)
            out[y0:y1, x0:x1] = sub
            # colour bleed into the rest of the transparent plate (clean bilinear sampling)
            out[..., :3] = np.median(sub[..., :3][sub[..., 3] > 0.5], axis=0) if (sub[..., 3] > 0.5).any() else 0
            out[y0:y1, x0:x1] = sub
            return out
    return _paint(puffs, w, h, sun, sun_z, pal, floor_rgb=floor_rgb, **kw)


def _paint(puffs, w, h, sun, sun_z=0.35, pal='noon', form_sigma=None, lobe=0.1, bands=4,
          band_soft=0.12, cap=0.75, cap_gain=0.6, body=0.05, form_w=0.3, crease=0.4, sub_cap=0.6, shade_heads=0.18, scallop=1.3, crest=0.0, crest_hdr=1.0, crest_drop=1.2, paint_r=3.5, paint_aniso=2.2, cap_plateau=0.25, cap_fall=1.6, brush=0.05, rim=1.0, rim_px=None, lost=0.45, translucency=0.35, air=None,
          floor_rgb=None, seed=0, wrap=0.35, value_bias=0.0,
          brush_angle=-20.0, sky_lift=0.25, bottom_dark=0.25, contrast=1.0, debug=None, floor_y=0.0, unit=None):
    """Paint a puff set into a straight-alpha RGBA plate (the painter's decisions, in order):

    1 big form      one 'pillow' per cloud from the silhouette distance field -> ONE terminator per
                    cloud: form_sigma (px, default ~ largest puff), wrap, form_w (its share of value),
                    lobe (0..1 floret-normal detail), body (base value of the unlit mass).
    2 light caps    head-aware march toward the sun: each primary head (and, multiplicatively, each
                    sub-head: sub_cap) gets a painted light shape on its sunward edge - crisp scalloped
                    outer edge, soft inner edge: cap (width, x head radius), cap_plateau / cap_fall
                    (flat-lit part / soft falloff, in cap widths), cap_gain, scallop (lit shapes bulge
                    into florets straddling the edge), shade_heads (heads still read on the shade side).
    3 crease        rear head's shaded underside right behind a front head's lit top edge: crease.
    4 values        bands soft-posterised steps with brush-modulated edge hardness (band_soft), dab
                    noise (brush, brush_angle), value_bias, contrast; colour from the palette ramp
                    deep..hi, sky_lift (up-facing shadow tint), bottom_dark (bounce underneath).
    5 edges         rim / rim_px: HDR lining on the sunlit outline; translucency (backlit edge glow);
                    crest / crest_hdr / crest_drop: gold crest line on every sunward top edge that
                    drops onto something well behind it (sea of clouds, backlit towers);
                    lost: shadow-side silhouette feathers into the air colour.
    6 painter pass  paint_r / paint_aniso: 8-sector Kuwahara -> flat brushy patches, strongest in shadow.
    7 air           per-puff haze toward `air` (default palette 'air'); per-puff 'val' value bias.
    sun: (x, y) plate px (a far point = directional light); sun_z: elevation toward the viewer
    (>0 front-lit faces, <0 backlit). floor_rgb (+floor_y): opaque plate, colour (or (h,w,3) image)
    under/between puffs below floor_y (sea valleys). unit: px per 1080p px (default w/1920).
    debug: dict to receive intermediate fields (cap, crease, form, lam).
    """
    pal = palette(pal)
    unit = unit or w / 1920.0
    air = pal['air'] if air is None else np.asarray(air, np.float32)
    X, Y, Z, R = (np.ascontiguousarray(puffs[k], np.float32) for k in ('x', 'y', 'z', 'r'))
    FL, FF, CE, HZ = (np.ascontiguousarray(puffs[k], np.float32) for k in ('floor', 'fade', 'ceil', 'haze'))
    SQ = np.ascontiguousarray(np.where(puffs['sq'] > 0, puffs['sq'], 1.0), np.float32)
    zb = np.full((h, w), -1e9, np.float32)
    nx = np.zeros((h, w), np.float32)
    ny = np.zeros((h, w), np.float32)
    nz = np.ones((h, w), np.float32)
    cov = np.zeros((h, w), np.float32)
    sid = np.zeros((h, w), np.int32)
    if len(X):
        _splat(X, Y, Z, R, FL, FF, CE, SQ, h, w, zb, nx, ny, nz, cov, sid)
    inside = cov > 0.01
    if form_sigma is None:
        form_sigma = float(np.percentile(R, 97)) * 0.9 if len(R) else 20.0
    xs, ys = C.grid(w, h)
    # ---- normals: whole-cloud form + per-head domes (crisp at head overlaps) + a little floret detail
    wgt = _blur(cov, form_sigma) + 1e-4
    Nf = [_blur(n * cov, form_sigma) / wgt for n in (nx, ny, nz)]
    H0 = puffs['h0'].astype(np.int64)
    H1 = puffs['h1'].astype(np.int64)

    def head_normal(hidx, flat):
        k = hidx[sid]
        rr = R[k] * flat
        a = (xs - X[k]) / rr
        b = (ys - Y[k]) / rr
        q = a * a + b * b
        sc = np.where(q > 1, 1 / np.sqrt(q + 1e-9), 1.0)
        a, b = a * sc, b * sc
        c = np.sqrt(np.clip(1 - a * a - b * b, 0, 1))
        re = R[k] * 1.4                    # enlarged dome: a head's florets belong to its surface
        zz = Z[k] + np.sqrt(np.clip(re ** 2 - ((xs - X[k]) ** 2 + (ys - Y[k]) ** 2), 0, None))
        return [a, b, c], zz, k

    _, zh1, k1 = head_normal(H1, 1.0)
    # big form = one pillow per cloud: height from the silhouette distance field -> normals
    din = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    ph = np.sqrt(np.minimum(din, form_sigma * 3) * form_sigma * 2.0)
    ph = cv2.GaussianBlur(_blur(ph, form_sigma * 0.35), (0, 0), 3)
    gxf = cv2.Sobel(ph, cv2.CV_32F, 1, 0, ksize=3) / 8
    gyf = cv2.Sobel(ph, cv2.CV_32F, 0, 1, ksize=3) / 8
    Nf = [-gxf, -gyf, np.ones_like(gxf)]
    ls = max(1.0, form_sigma * 0.04)
    Nl = [_blur(n, ls) for n in (nx, ny, nz)]
    nn = np.sqrt(Nf[0] ** 2 + Nf[1] ** 2 + 1) + 1e-6
    N = [Nf[i] / nn * (1 - lobe) + Nl[i] * lobe for i in range(3)]
    nn = np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2) + 1e-6
    N = [n / nn for n in N]
    # ---- light direction per pixel (toward the sun position)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-6
    lxy = math.sqrt(max(1 - sun_z * sun_z, 0))
    Lx, Ly = dx / dl * lxy, dy / dl * lxy
    ndl = N[0] * Lx + N[1] * Ly + N[2] * sun_z
    form = np.clip((ndl + wrap) / (1 + wrap), 0, 1)          # big-form terminator
    # ---- light caps: painted on the sunward edge of every head (scalloped outer edge, soft inner edge)
    sa = math.atan2(sun[1] - h / 2, sun[0] - w / 2)
    SX, SY = np.float32(sun[0]), np.float32(sun[1])
    cn = _noise(w, h, w / (48.0 * unit), seed + 31, octaves=3) - 0.5

    def cross_kernel(length):
        klen = int(max(3, length)) | 1
        ker = np.zeros((klen, klen), np.float32)
        cc = klen // 2
        px, py = -math.sin(sa), math.cos(sa)
        for tt in np.linspace(-cc, cc, klen * 2):
            ker[int(round(cc + py * tt)), int(round(cc + px * tt))] += math.exp(-(tt / (cc * 0.6 + 1e-6)) ** 2)
        return ker / ker.sum()

    # position inside the winning floret (1 centre .. 0 edge): lit shapes bulge into each floret
    fx_ = xs - X[sid]
    fy_ = (ys - Y[sid]) / SQ[sid]
    bul = np.sqrt(np.clip(1 - np.sqrt(fx_ * fx_ + fy_ * fy_) / R[sid], 0, 1)) * inside
    Rs = R[sid]

    def cap_pass(HC, capk, plateau):
        hid = np.ascontiguousarray(HC[sid]).astype(np.int64)
        rh = np.where(inside, R[hid], 1.0).astype(np.float32)
        capw = (rh * capk).astype(np.float32)
        dzo = np.zeros((h, w), np.float32)
        d = _caps(zb, cov, hid, X, Y, Z, R, capw, SX, SY, np.float32(3.0), np.float32(1.2), 0, dzo)
        dn = (d - scallop * bul * Rs * 0.6) / capw + cn * 0.5
        cl = 1 - _ss(plateau, plateau + cap_fall, dn)
        cl = cl * 0.85 + 0.15 * np.exp(-dn / 2.5)
        ker = cross_kernel(float(np.median(capw[inside])) * 0.6 if inside.any() else 3)
        cl = cv2.filter2D(cl * cov, -1, ker) / (cv2.filter2D(cov, -1, ker) + 1e-4)
        cl = cv2.GaussianBlur(cl, (0, 0), max(0.7, form_sigma * 0.006)) * cov * cov
        return cl, hid, capw, ker, d, dzo

    capl, _, capw0, _, d0, dz0 = cap_pass(H0, cap, cap_plateau)
    cap1, hid1, capw1, ker1, d1, _ = cap_pass(H1, cap * 0.8, 0.35)
    cr = _caps(zb, cov, hid1, X, Y, Z, R, capw1, SX, SY, np.float32(1.3), np.float32(0.0), 1, np.zeros((h, w), np.float32))
    cr = cv2.filter2D(cr, -1, ker1)
    cr = cv2.GaussianBlur(cr, (0, 0), max(0.8, form_sigma * 0.008))
    capl = capl * (1 - sub_cap + sub_cap * cap1)            # sub-heads model the inside of lit heads
    lam = body + (1 - body) * form * form_w + capl * (0.2 + 0.8 * form ** 1.5) * cap_gain
    lam = lam + cap1 * shade_heads * (1 - form)          # heads still read on the shade side (sky light)
    lam = lam * (1 - crease * cr) + puffs['val'][sid] * inside
    if debug is not None:
        debug.update(cap=capl, crease=cr, form=form, lam=lam.copy())
    lam = np.clip(lam, 0, 1.2)
    # ---- silhouette geometry: distance inside, outward normal, sun facing
    dist = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    ca = cv2.GaussianBlur(cov, (0, 0), 2.5)
    gx = cv2.Sobel(ca, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ca, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    ox, oy = -gx / gl, -gy / gl
    Lxn, Lyn = dx / dl, dy / dl
    facing = ox * Lxn + oy * Lyn            # +1: outline faces the sun
    rim_px = rim_px or max(1.5, unit * 3.0)
    band = np.exp(-dist / rim_px)
    # ---- painted value: light -> soft posterised bands with brush-modulated edges
    dab = _noise(w, h, w / (21.0 * unit), seed + 11, octaves=3, angle=brush_angle, stretch=3.0) - 0.5
    hard = _noise(w, h, w / (270.0 * unit), seed + 23, octaves=2)
    v = lam * contrast + value_bias + (1 - contrast) * 0.5
    v = v + dab * brush * 2
    # thin edges are brighter (light bleeds through the outline)
    trans = band * np.clip(0.5 + 0.5 * facing, 0, 1)
    v = v + trans * (0.22 if sun_z >= 0 else 0.1)
    if bands > 0:
        q = np.zeros_like(v)
        soft = band_soft * (0.25 + 1.5 * hard)
        for k in range(bands):
            t = (k + 0.5) / bands
            q += _ss(t - soft, t + soft, v)
        q /= bands
        v = q * 0.55 + np.clip(v, 0, 1) * 0.45
    v = np.clip(v, 0, 1)
    col = _ramp(v, pal)
    # sky light on up-facing shadowed areas; bounce/dark underside
    up = np.clip(-N[1], 0, 1)
    down = np.clip(N[1], 0, 1)
    shade_w = (1 - v)[..., None]
    col = col + (pal['sky'] - col) * (sky_lift * up * (1 - v))[..., None]
    col = col + (pal['bounce'] - col) * (bottom_dark * down * (1 - v) * 0.6)[..., None]
    # ---- rim / lining (HDR) on the sun-facing outline where lit
    litedge = np.clip(facing * 1.2 + 0.1, 0, 1)
    if sun_z < 0:  # backlit: the whole outline glows, strongest toward the sun
        litedge = np.clip(0.35 + 0.65 * facing, 0, 1)
    rimk = rim * np.exp(-dist / (rim_px * 0.6)) * litedge * np.clip(lam * 1.6 + (0.6 if sun_z < 0 else 0.1), 0, 1)
    col = col + (pal['rim'] - col) * np.clip(rimk, 0, 1)[..., None]
    if crest > 0:
        # painted crest line: every head's sunward top edge (interior overlaps too), hotter near the sun
        # axis; thin core + soft inner falloff
        rp = rim_px or max(1.5, unit * 3.0)
        dm = d0
        cl = np.exp(-dm / (rp * 0.9)) * 0.8 + np.exp(-dm / (rp * 3.0)) * 0.35
        axis = 0.35 + 0.65 * np.exp(-((xs - sun[0]) / (0.45 * w)) ** 2)
        # only where the ridge drops onto something well behind it (row edges), not between neighbours
        big = np.clip(dz0 / (capw0 / max(cap, 1e-3) * crest_drop + 1e-3), 0, 1)
        big = cv2.dilate(big, np.ones((3, 3), np.uint8))
        cl = cv2.GaussianBlur((cl * big).astype(np.float32), (0, 0), 0.6) * axis * crest * inside
        ccol = pal['rim'] * 0.5 + pal['lit'] * 0.5
        col = col + (ccol * crest_hdr - col) * np.clip(cl, 0, 1)[..., None]
    if sun_z < 0 and translucency > 0:
        tr = np.exp(-dist / (rim_px * 4)) * np.clip(facing, 0, 1) * translucency
        col = col + pal['lit'] * tr[..., None] * 0.6
    # ---- painter's pass: flat brushy value patches, strongest in shadow (detail is lost in the dark)
    if paint_r > 0:
        pr = paint_r * unit
        kw_ = kuwahara(col, pr, aniso=paint_aniso, angle=brush_angle)
        kw2 = kuwahara(col, pr * 0.5)
        sh_w = np.clip(1.15 - v * 1.3, 0.25, 1.0)[..., None] * inside[..., None]
        col = col * (1 - sh_w) + (kw_ * 0.6 + kw2 * 0.4) * sh_w
    # ---- aerial perspective per puff
    hz = HZ[sid] * inside
    col = col + (air - col) * hz[..., None]
    # ---- lost edges: shadow-side / underside silhouette dissolves toward air
    lostw = lost * np.clip(1 - v * 1.6, 0, 1) * np.clip(-facing, 0, 1)
    fe = np.clip(dist / (rim_px * 3), 0, 1) ** 0.7
    alpha = cov * (1 - lostw * (1 - fe))
    col = col + (air - col) * (lostw * (1 - fe) * 0.7)[..., None]
    # ---- bleed colour into the transparent margin (clean bilinear drift sampling)
    if floor_rgb is not None:
        col = np.where(inside[..., None], col, np.asarray(floor_rgb, np.float32))
        below = _ss(floor_y - 1.5, floor_y + 1.5, ys)
        alpha = np.maximum(alpha, below)
        m = (alpha > 0.3).astype(np.float32)
        acc = cv2.GaussianBlur(col * m[..., None], (0, 0), 3) / (cv2.GaussianBlur(m, (0, 0), 3)[..., None] + 1e-4)
        col = np.where((alpha > 0.3)[..., None], col, acc)
    else:
        m = (cov > 0.3).astype(np.float32)
        acc = cv2.GaussianBlur(col * m[..., None], (0, 0), 3) / (cv2.GaussianBlur(m, (0, 0), 3)[..., None] + 1e-4)
        col = np.where((cov > 0.3)[..., None], col, acc)
    out = np.dstack([col, alpha]).astype(np.float32)
    out[..., 3] = np.clip(out[..., 3], 0, 1)
    return out


# ============================================================================== plates

def cumulonimbus_plate(w, h, cx, base_y, height, sun, sun_z=0.35, pal='noon', seed=0, width=None, anvil=0.0,
                       turrets=3, lean=0.0, haze=0.0, anvil_dir=1.0, levels=3, **kw):
    width = width or height * 0.75
    P = tower_puffs(cx, base_y, height, width, seed=seed, lean=lean, turrets=turrets, anvil=anvil,
                    anvil_dir=anvil_dir, levels=levels, haze=haze)
    kw.setdefault('form_sigma', width * 0.18)
    return paint(P, w, h, sun, sun_z, pal, seed=seed, **kw)


def cumulus_plate(w, h, clouds, sun, sun_z=0.35, pal='noon', levels=3, **kw):
    ps = []
    for i, c in enumerate(clouds):
        cx, by, cw, ch, sd = c[:5]
        hz = c[5] if len(c) > 5 else 0.0
        ps.append(heap_puffs(cx, by, cw, ch, seed=sd, levels=levels, haze=hz, z0=-i * ch * 4))
    P = merge(*ps)
    kw.setdefault('form_sigma', float(np.median([c[3] for c in clouds])) * 0.45)
    return paint(P, w, h, sun, sun_z, pal, **kw)


def horizon_bank_plate(w, h, y, height, sun, sun_z=0.35, pal='noon', seed=0, rows=2, haze=0.45, flat=0.8,
                       spread=(1.6, 3.6), **kw):
    """Low cloud bank / flat stratocumulus shelf along y (use flat~0.95, wide spread and value_bias<0 for
    the dark shelf under a tower)."""
    P = bank_puffs(-0.02 * w, 1.02 * w, y, height, seed=seed, rows=rows, haze=haze, flat=flat, spread=spread)
    kw.setdefault('form_sigma', height * 0.5)
    return paint(P, w, h, sun, sun_z, pal, seed=seed, **kw)


def sea_of_clouds_plate(w, h, horizon_y, sun, sun_z=-0.3, pal='sunrise', seed=0, splits=(), cam_h=2.5,
                        relief=0.6, cell=0.9, floor_rgb=None, valley=0.25, air=None, haze_far=0.85, far=110.0, plate_kw=None, **kw):
    """Sea of clouds from above. Returns a list of RGBA plates far -> near, split at world depths
    `splits` (descending order not required); the farthest plate is opaque (valley floor = deep).
    Composite sea_tower plates between them (a tower at depth D goes after the plate that ends at D).
    plate_kw: optional list (far -> near) of per-plate paint() overrides, e.g. make the nearest plate
    softer/darker (painters lose detail in the foreground: paint_r=16, paint_aniso=4, value_bias=-0.1)."""
    palp = palette(pal)
    V = sea_view(w, h, horizon_y, cam_h)
    edges = sorted(set([None] + list(splits)), key=lambda v: -1 if v is None else v)
    ranges = []
    lo = None
    for e in sorted(splits):
        ranges.append((lo, e))
        lo = e
    ranges.append((lo, far))
    ranges = ranges[::-1]                                   # far -> near
    air_c = palp['air'] if air is None else np.asarray(air, np.float32)
    if floor_rgb is None:
        floor_rgb = palp['deep'] * 1.05 + palp['shade'] * 0.1
    yy = np.arange(h, dtype=np.float32)[:, None, None]
    tf = np.clip((yy - horizon_y) / (0.3 * (h - horizon_y)), 0, 1) ** 1.3
    floor_img = (air_c * (1 - tf) + np.asarray(floor_rgb, np.float32) * tf) * np.ones((1, w, 1), np.float32)
    kw.setdefault('lost', 0.0)
    kw.setdefault('form_sigma', w * 0.03)
    if sun_z < 0:                       # backlit sea: thin hot crests on every sunward top edge, dark bodies
        kw.setdefault('cap', 0.18)
        kw.setdefault('cap_gain', 0.95)
        kw.setdefault('form_w', 0.18)
        kw.setdefault('body', 0.06)
        kw.setdefault('sub_cap', 0.3)
        kw.setdefault('shade_heads', 0.1)
        kw.setdefault('scallop', 0.35)
        kw.setdefault('value_bias', -0.06)
        kw.setdefault('sky_lift', 0.12)
        kw.setdefault('crest', 0.9)
        kw.setdefault('crest_hdr', 1.15)
        kw.setdefault('cap_plateau', 0.55)
        kw.setdefault('cap_fall', 0.8)
        kw.setdefault('crease', 0.6)
        kw.setdefault('sub_cap', 0.3)
        kw.setdefault('band_soft', 0.08)
        kw.setdefault('paint_r', 6.0)
    out = []
    plate_kw = plate_kw or [{}] * len(ranges)
    for i, (a, b) in enumerate(ranges):
        kk = dict(kw)
        kk.update(plate_kw[i] if i < len(plate_kw) else {})
        P = sea_puffs(w, h, horizon_y, seed=seed, cam_h=cam_h, d_range=(a, b), cell=cell, relief=relief,
                      valley=valley, haze_far=haze_far, far_ref=far)
        # only the far plate is opaque
        out.append(paint(P, w, h, sun, sun_z, pal, floor_rgb=floor_img if i == 0 else None, floor_y=horizon_y,
                         seed=seed + i, air=air, **kk))
    return out


def sea_tower_plate(w, h, horizon_y, x, depth, height, sun, sun_z=0.0, pal='sunrise', seed=0, cam_h=2.5,
                    width=0.6, turrets=3, anvil=0.0, lean=0.0, haze=0.0, **kw):
    """Painted tower standing in the sea (see sea_tower_puffs); RGBA plate."""
    P = sea_tower_puffs(w, h, horizon_y, x, depth, height, seed=seed, cam_h=cam_h, width=width, turrets=turrets,
                        anvil=anvil, lean=lean, haze=haze)
    kw.setdefault('form_sigma', float(np.percentile(P['r'], 97)) * 0.9)
    return paint(P, w, h, sun, sun_z, pal, seed=seed, **kw)


def cirrus_plate(w, h, sun, pal='noon', seed=0, region=(0.02, 0.45), angle=-12.0, density=0.5,
                 mackerel=0.5, opacity=0.8, color=None, fleck=None, streaks=1.0):
    """High cloud painted as brush marks (gow_02 / wwy_01 / cm5_04):
    * mackerel flecks - small elongated dabs (fleck px at 1080p ~ 4-18) arranged in rippled rows
      along the flow `angle`, in drifting patches; each dab = cool base + a brighter part offset
      toward the sun (the painters' two-value fleck);
    * cirrus strokes - long tapered, broken strokes along the flow.
    region: (top, bottom) fraction of h where the high cloud lives (fading out at both ends)."""
    pal = palette(pal)
    rng = np.random.default_rng(seed)
    u = w / 1920.0
    fleck = fleck or 5.5 * u
    th = math.radians(angle)
    fdx, fdy = math.cos(th), math.sin(th)
    sx, sy = sun[0] - w / 2, sun[1] - h / 2
    sl = math.hypot(sx, sy) + 1e-6
    lx, ly = sx / sl, sy / sl
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    reg = _ss(region[0] - 0.05, region[0] + 0.1, ys) * (1 - _ss(region[1] - 0.15, region[1], ys))
    reg = np.broadcast_to(reg, (h, w))
    patch = _noise(w, h, 2.2, seed + 3, 3, angle=angle, stretch=2.5)
    xs_, ys_, as_, bs_, an_, op_ = [], [], [], [], [], []
    # ---- mackerel dabs
    if mackerel > 0:
        n = int(14000 * mackerel * (w * h) / (1920 * 1080))
        px = rng.uniform(0, w, n * 3)
        py = rng.uniform(0, h, n * 3)
        ix = np.clip(px.astype(int), 0, w - 1)
        iy = np.clip(py.astype(int), 0, h - 1)
        dens = patch[iy, ix] * reg[iy, ix]
        # rippled rows perpendicular to the flow
        perp = (-px * fdy + py * fdx) / (fleck * 3.2)
        along = (px * fdx + py * fdy) / (fleck * 9.0)
        rip = 0.5 + 0.5 * np.sin(perp * 2 * math.pi + 0.8 * np.sin(along * 1.3))
        keep = rng.random(n * 3) < np.clip((dens - (0.62 - 0.25 * mackerel)) * 4.5, 0, 1) * (0.25 + 0.75 * rip)
        px, py, dd = px[keep][:n], py[keep][:n], dens[keep][:n]
        # clusters: each fleck is 1-4 overlapping sub-dabs strung along the flow
        reps = rng.integers(1, 5, len(px))
        px = np.repeat(px, reps) + rng.normal(0, 1, reps.sum()) * fleck * 1.3 * fdx
        py = np.repeat(py, reps) + rng.normal(0, 1, reps.sum()) * fleck * 1.3 * fdy + rng.normal(0, fleck * 0.3, reps.sum())
        dd = np.repeat(dd, reps)
        m = len(px)
        size = fleck * rng.uniform(0.4, 1.2, m) * (0.5 + dd)
        xs_.append(px); ys_.append(py)
        as_.append(size * rng.uniform(1.4, 2.6, m)); bs_.append(size * rng.uniform(0.5, 0.8, m))
        an_.append(th + rng.normal(0, 0.3, m)); op_.append(np.clip(0.5 + 0.6 * dd, 0, 1) * rng.uniform(0.7, 1.0, m))
    # ---- cirrus strokes (chains of overlapping dabs, tapered at both ends)
    if streaks > 0 and density > 0:
        ns = int(14 * density * streaks * w / 1920)
        for k in range(ns):
            for _ in range(20):
                x0 = rng.uniform(-0.1 * w, 1.1 * w)
                y0 = rng.uniform(0, h)
                if reg[int(np.clip(y0, 0, h - 1)), int(np.clip(x0, 0, w - 1))] > rng.uniform(0.2, 1.0):
                    break
            L = rng.uniform(0.08, 0.35) * w
            wd = rng.uniform(3.0, 14.0) * u * (1 + 1.5 * density)
            curv = rng.normal(0, 0.35)
            npts = int(L / max(wd * 0.5, 1.0))
            t = np.linspace(0, 1, npts)
            ang = th + curv * (t - 0.5) + rng.normal(0, 0.05)
            px = x0 + np.cumsum(np.cos(ang)) * (L / npts)
            py = y0 + np.cumsum(np.sin(ang)) * (L / npts)
            prof = np.sin(np.pi * t) ** 0.7
            brk = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(t * rng.uniform(6, 18) + rng.uniform(0, 6)))
            xs_.append(px); ys_.append(py)
            as_.append(np.full(npts, wd * 1.6)); bs_.append(np.maximum(wd * prof, 0.6))
            an_.append(ang); op_.append(prof * brk * rng.uniform(0.2, 0.5))
    if not xs_:
        return np.zeros((h, w, 4), np.float32)
    X = np.concatenate(xs_).astype(np.float32); Y = np.concatenate(ys_).astype(np.float32)
    A = np.maximum(np.concatenate(as_), 0.8).astype(np.float32); B = np.maximum(np.concatenate(bs_), 0.6).astype(np.float32)
    AN = np.concatenate(an_).astype(np.float32); OP = np.concatenate(op_).astype(np.float32)
    off = (B * 0.45).astype(np.float32)
    alpha = np.zeros((h, w), np.float32)
    lit = np.zeros((h, w), np.float32)
    _ellipses(X, Y, A, B, AN, OP, (lx * off).astype(np.float32), (ly * off).astype(np.float32),
              np.full(len(X), 0.8, np.float32), h, w, alpha, lit)
    # ragged dry-brush edges: displace the marks with fine noise, break them with speckle
    dxn = (_noise(w, h, w / (fleck * 3.0), seed + 41, 2) - 0.5) * fleck * 1.6
    dyn = (_noise(w, h, w / (fleck * 3.0), seed + 43, 2) - 0.5) * fleck * 1.6
    alpha = C.warp(alpha, dxn, dyn, cv2.BORDER_CONSTANT)
    lit = C.warp(lit, dxn, dyn, cv2.BORDER_CONSTANT)
    sp = _noise(w, h, w / (fleck * 0.9), seed + 47, 2)
    alpha = alpha * (0.55 + 0.45 * _ss(0.25, 0.6, sp))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6 * max(u, 0.5))
    base = pal['sky'] * 0.35 + pal['lit'] * 0.65 if color is None else np.asarray(color, np.float32) * 0.8
    top = pal['hi'] if color is None else np.asarray(color, np.float32)
    col = base + (top - base) * lit[..., None]
    a = alpha * opacity
    m = (a > 0.02).astype(np.float32)
    acc = cv2.GaussianBlur(col * m[..., None], (0, 0), 2) / (cv2.GaussianBlur(m, (0, 0), 2)[..., None] + 1e-4)
    col = np.where(m[..., None] > 0, col, acc)
    return np.dstack([col, np.clip(a, 0, 1)]).astype(np.float32)


def stratus_band(w, h, y, thick, sun, pal='noon', seed=0, color=None, lit_color=None, opacity=0.9, layers=3,
                 x0=None, x1=None):
    """Flat layered stratus band (the dark shelf under wwy_01's towers / cm5_06's dusk bars): long
    horizontal strokes stacked in `layers`, cool body, a thin lit upper edge, ragged ends."""
    pal = palette(pal)
    rng = np.random.default_rng(seed)
    u = w / 1920.0
    x0 = -0.05 * w if x0 is None else x0
    x1 = 1.05 * w if x1 is None else x1
    xs_, ys_, as_, bs_, an_, op_ = [], [], [], [], [], []
    for L in range(layers):
        yl = y - thick * 0.5 * L
        n = int((x1 - x0) / (0.08 * w)) + 3
        for k in range(n):
            cx = rng.uniform(x0, x1)
            ln = rng.uniform(0.1, 0.4) * w
            wd = thick * rng.uniform(0.25, 0.6) * (1 - 0.25 * L)
            npts = max(int(ln / max(wd * 0.6, 1.0)), 3)
            t = np.linspace(0, 1, npts)
            px = cx - ln / 2 + t * ln
            py = yl + rng.normal(0, 0.15) * thick + np.sin(t * rng.uniform(2, 5) + rng.uniform(0, 6)) * wd * 0.2
            prof = np.sin(np.pi * t) ** 0.35
            xs_.append(px); ys_.append(py); as_.append(np.full(npts, wd * 2.2)); bs_.append(np.maximum(wd * prof, 0.8))
            an_.append(np.zeros(npts)); op_.append(prof * rng.uniform(0.7, 1.0))
    X = np.concatenate(xs_).astype(np.float32); Y = np.concatenate(ys_).astype(np.float32)
    A = np.concatenate(as_).astype(np.float32); B = np.concatenate(bs_).astype(np.float32)
    AN = np.concatenate(an_).astype(np.float32); OP = np.concatenate(op_).astype(np.float32)
    alpha = np.zeros((h, w), np.float32)
    lit = np.zeros((h, w), np.float32)
    _ellipses(X, Y, A, B, AN, OP, np.zeros_like(X), (-B * 0.55).astype(np.float32), np.full(len(X), 0.75, np.float32),
              h, w, alpha, lit)
    dxn = (_noise(w, h, 60 * u, seed + 5, 2) - 0.5) * thick * 0.4
    dyn = (_noise(w, h, 60 * u, seed + 6, 2) - 0.5) * thick * 0.25
    alpha = C.warp(alpha, dxn, dyn, cv2.BORDER_CONSTANT)
    lit = C.warp(lit, dxn, dyn, cv2.BORDER_CONSTANT)
    rim = np.clip(alpha - cv2.GaussianBlur(lit, (0, 0), 1.0), 0, 1)       # thin upper edge lit
    body = pal['shade'] * 0.7 + pal['mid'] * 0.3 if color is None else np.asarray(color, np.float32)
    lc = pal['lit'] if lit_color is None else np.asarray(lit_color, np.float32)
    top = cv2.GaussianBlur(np.clip(alpha - _shift(lit, 0, -2 * u), 0, 1), (0, 0), 0.7)
    col = body + (pal['mid'] - body) * (0.5 * lit)[..., None] + (lc - body) * np.clip(top * 1.2, 0, 1)[..., None]
    m = (alpha > 0.02).astype(np.float32)
    acc = cv2.GaussianBlur(col * m[..., None], (0, 0), 2) / (cv2.GaussianBlur(m, (0, 0), 2)[..., None] + 1e-4)
    col = np.where(m[..., None] > 0, col, acc)
    return np.dstack([col, np.clip(alpha * opacity, 0, 1)]).astype(np.float32)


def _shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), borderMode=cv2.BORDER_CONSTANT)


# ============================================================================== drift / compositing

def drift(plate, W, H, t, speed=(0.003, 0.0), cam=(0.0, 0.0), zoom=1.0, depth=1.0, billow=0.0, seed=0,
          billow_scale=0.03, billow_rate=0.3):
    """Sample a plate into a W x H frame: wind drift (fraction of W per s) + camera parallax + zoom,
    optional slow coherent billowing. See lib.sky.drift."""
    return _S.drift(plate, W, H, t, speed=speed, cam=cam, zoom=zoom, depth=depth, billow=billow,
                    billow_scale=billow_scale, billow_rate=billow_rate, seed=seed)


def screen_pos(p, W, H, plate_shape, cam=(0.0, 0.0), zoom=1.0, depth=1.0, t=0.0, speed=(0.0, 0.0)):
    """Where plate point p lands on screen after drift(...) with the same arguments."""
    h, w = plate_shape[:2]
    if not isinstance(speed, (tuple, list)):
        speed = (float(speed), 0.0)
    z = 1 + (zoom - 1) * depth
    cx = w / 2 - speed[0] * W * t + cam[0] * depth
    cy = h / 2 - speed[1] * W * t + cam[1] * depth
    return (W / 2 + (p[0] - cx) * z, H / 2 + (p[1] - cy) * z)


def over(img, *layers):
    for L in layers:
        img = _F.over_rgba(img, L)
    return img


def halo(layer, strength=0.25, sigma=None, threshold=1.0):
    """Additive halation from the HDR part (rim/lining) of a sampled RGBA layer: light spilling over
    the silhouette into the sky. Computed at 1/4 resolution. Returns (H, W, 3)."""
    H, W = layer.shape[:2]
    sigma = sigma or W * 0.006
    q = 4
    sm = cv2.resize(layer, (max(W // q, 2), max(H // q, 2)), interpolation=cv2.INTER_AREA)
    a = sm[..., 3:4]
    x = np.clip(sm[..., :3] - threshold, 0, None) * a
    x = x + np.clip(sm[..., :3].mean(-1, keepdims=True) - 0.85, 0, None) * a * 0.3
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), max(sigma / q, 0.5))
    return cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) * strength
