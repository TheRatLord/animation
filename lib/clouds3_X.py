"""clouds3_X - painted Shinkai-style clouds, round 3 (lobe volumes + painted value ramp).

WHY A NEW MODEL
---------------
Round 2 (lib/clouds2) painted clouds as stacks of flat 2D masses. Its known problems (ghosted, translucent
inner masses; a washed-out cream lit core; a flat one-value shadow; vertical stripe shading on towers;
outlined "sticker" lobes; repeating scallop rows; same-size edge lobes; flat bases) all come from the
same thing: the painting had no single form underneath it. Here every cloud is a 3D cluster of
ellipsoid LOBES, and the painting is derived from that cluster:

  * GEOMETRY - a few big core lobes fill a designed envelope (dome, tower, anvil, bank, sea deck).
    Florets are then grown fractally on the exposed upper / sunward surface. Child sizes follow a
    power law and parents get children in clumps, so the silhouette has big heads, medium florets and
    tiny crisp buds with bare smooth passages between them. Bases are cut by a (slightly jittered) flat
    plane, and florets never grow downward.
  * RASTER - every pixel ray-casts the lobe cluster analytically, with 2x supersampling. That gives a
    front depth, a back depth (the thickness of the solid), a normal and the id of the front lobe. The
    cloud is fully opaque inside the silhouette, so nothing is ghosted.
  * LIGHT - a volumetric screen-space shadow march: optical depth toward the sun through the solid
    between front and back depth. Upper lobes throw real shadows on the lobes under them. The
    terminator is soft where light penetrates and firm where a lobe shades its neighbour. Lambert
    shading uses a normal blended between the lobe's own normal and a heavily blurred "big form"
    normal, so each lobe reads as part of one mass, not as a clay ball.
  * PAINT - the light value is banded into 4 broad painted steps with noise-warped (brushy)
    boundaries and mapped through a palette ramp:
    shadow (sky-fill blue on up-facing, warm bounce below, deep indigo in the core)
    -> terminator (pink / violet at sunset) -> lit_lo -> lit -> hi.
    Thin sun-facing silhouettes get a silver/gold lining; backlit edges glow with forward scattering.
    Lit silhouettes stay crisp. Shadow-side and base silhouettes are "lost" (softened into the sky).
    Aerial perspective is applied by depth.

All colours are display RGB ~0..1 (hi / rim may exceed 1 for bloom). Plates are straight-alpha RGBA
float32 (h, w, 4), resolution independent (pass sizes in plate px, scale with W), deterministic per seed.
Plates bleed colour into transparent margins, so bilinear drift sampling never makes dark fringes.

API
===
Palettes
    PRESETS: 'noon', 'magic_hour', 'sunset', 'sunrise' (+ 'summer' = 'noon').
        keys: hi lit lit_lo mid shade deep sky bounce rim haze
    palette(p)                name | dict -> dict of float32 RGB arrays
    mix_palette(a, b, t)
Sun
    Side-view plates take sun=(x, y) (plate px, may be off-plate) or sun_dir=(dx, dy) (screen, y down),
    plus sun_z = how much the light comes toward the viewer (0.15 default = side / three-quarter light
    that keeps a readable terminator; 0.35 = frontal; < 0 = backlit). sea_of_clouds_plate uses a
    physical sun at infinity placed where it appears on screen (sun=(x, y)), + `lift` (painter's cheat:
    tops catch a low sun) and `sun_z`.
Plates (one call -> finished straight-alpha RGBA plate, float32 (h, w, 4))
    cumulonimbus_plate(w, h, cx, base_y, height, sun|sun_dir, preset='noon', seed=0, sun_z=0.15,
                       width=0.62*height, anvil=True, anvil_dir=1, anvil_len=0.55, lean=0.0, haze=0.0,
                       detail=1.0, ss=1.5, eye_y=None, persp=4.0, kind='tcu', turrets=3, base_fade=0.06, **paint)
        towering cumulus group (main column + `turrets` lower towers) with optional flat-topped anvil.
    cumulus_plate(w, h, clouds=[(cx, base_y, width, height, seed[, haze]), ...], sun|sun_dir,
                  preset='noon', sun_z=0.15, detail=1.0, ss=1.5, base_fade=0.12, **paint)
        fair-weather heaps, all rendered in ONE pass (later entries nearer).
    horizon_bank_plate(w, h, y, height, sun|sun_dir, preset='noon', seed=0, rows=3, haze=0.35,
                       sun_z=0.2, ss=1.5, detail=0.8, **paint)      low receding bank of heaps
    sea_of_clouds_plate(w, h, horizon_y, sun=(x, y), preset='sunrise', seed=0, towers=[dict(x=screen x,
                        dist=distance in cam heights, height, width (x height), seed, lean, anvil, kind,
                        warm)], fov=55, near_r=0.06, grow_exp=0.72, far_dist=45, valley=0.55,
                        valley_depth=2.2, relief=0.35, detail=1.0, lift=0.05, sun_z=0.0, ss=1.25,
                        fog=0.008, horizon_haze=(colour, amount), glow=0.5, crest_width=0.3,
                        split=None, **paint)
        -> opaque-below-horizon RGBA plate; with split=d (distance in cam heights) returns
        [far_plate, near_plate] for parallax (the near plate is shaded with the far lobes present, so
        its shadows stay correct).
    cirrus_plate(w, h, preset, seed, region=(0.05, 0.4), angle=-8, density=0.5, opacity=0.6)
    cloudlets_plate(w, h, preset, seed, region=(0.05, 0.45), density=0.5, size=0.012, angle=-12,
                    stretch=2.2, opacity=0.85, sun_dir=(0.6, -0.8), patch=0.35)   mackerel sky flecks
    haze_plate(w, h, y0, y1, color, amount=0.6, y2=None)
    dissolve_base(plate, cx, half_w, base_y, fade_h, seed)   torn, lost base (in place)
Painting controls (**paint, forwarded to render_lobes; defaults tuned for Shinkai day clouds)
    form (0.55) big-form normal weight, head (0.6) head-normal weight, micro (0.06) floret normals,
    wrap (0.35), density (2.2) self-shadow, shadow_soft (0.35), contrast (0.45) / pivot (0.3) lit-vs-
    shadow simplification, bands (0.6) painted value steps, band_noise, warp (brushy boundaries),
    edge (1.0) crisp lit lobe-top / crest bands, crest_focus=(x, width, floor) (crests hottest near the
    sun axis), limb (0.25) white kept for the sunward silhouette, rim / rim_px lining, glow backlit
    forward scatter, dimple (1.0) fills AO-like pits, paint (1.0) Kuwahara brush pass, dab_amt (0.025),
    lost (1.0) soft shadow-side silhouettes, base_soft, haze / fog aerial perspective, keep (mask).
Low level
    Lobes()                    ellipsoid lobes (camera space) with head links: .add(c, r, cut_hi, cut_lo,
                               scale, haze, warm, root, dark), .add_grown(), .extend(), .array() (N, 13)
    cloud_lobes(rng, X, Yb, Z, width, height, kind='cu'|'tcu'|'cb'|'bank', ...)   one cloud's lobes
    surface_grow(rng, lobes, sizes, dens, ...)   multi-scale cauliflower growth on the exposed surface
    render_lobes(lobes, w, h, f, ppx, ppy, L, pal, ss=2, ...) -> RGBA   (the painter)
    kuwahara(img, r, q)        generalized Kuwahara (painterly flat patches)
Drift / compositing helpers (same contract as clouds2)
    drift(plate, W, H, t, speed, cam, zoom, depth, billow=0, billow_scale, billow_rate, seed)
    screen_pos(p, W, H, plate_size, cam, zoom, depth, t, speed)
    occluder(*layers), composite(img, *layers)
Cost at 1080p (warm numba cache; the first run after editing this file adds ~8-10 s of JIT compile):
    cumulonimbus_plate ~7-9 s, sea_of_clouds_plate with split ~12-15 s, cumulus / bank ~2-4 s;
    the demo scenes set up in ~15-20 s and render ~1-1.5 s / frame.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C
from . import sky as _S
from . import fx as _F

F32 = np.float32
BIG = 1e30
_DEBUG = None

# ----------------------------------------------------------------------------------------- palettes


def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, F32)


# hi: sun-facing peak (HDR ok)   lit: main lit face   lit_lo: lit face turning away   mid: terminator band
# shade: shadow lit by the sky   deep: shadow core / valleys   sky: sky-fill tint on up-facing shadow
# bounce: warm light from below on down-facing shadow   rim: lining (HDR)   haze: aerial perspective
PRESETS = {
    'noon': dict(hi=(1.04, 1.03, 0.97), lit=(0.96, 0.96, 0.93), lit_lo=(0.8, 0.87, 0.92), mid=(0.64, 0.76, 0.88),
                 shade=(0.48, 0.63, 0.83), deep=(0.33, 0.48, 0.73), sky=(0.66, 0.8, 0.93),
                 bounce=(0.55, 0.66, 0.82), rim=(1.35, 1.33, 1.22), haze=(0.72, 0.85, 0.97)),
    'magic_hour': dict(hi=(1.06, 0.96, 0.8), lit=(1.0, 0.86, 0.66), lit_lo=(0.94, 0.72, 0.66), mid=(0.86, 0.6, 0.7),
                       shade=(0.6, 0.58, 0.8), deep=(0.36, 0.34, 0.62), sky=(0.62, 0.66, 0.9),
                       bounce=(0.9, 0.7, 0.66), rim=(1.4, 1.15, 0.8), haze=(0.94, 0.8, 0.76)),
    'sunset': dict(hi=(1.08, 0.86, 0.6), lit=(1.0, 0.68, 0.45), lit_lo=(0.94, 0.54, 0.54), mid=(0.84, 0.44, 0.6),
                   shade=(0.5, 0.42, 0.7), deep=(0.22, 0.18, 0.44), sky=(0.52, 0.5, 0.82),
                   bounce=(0.88, 0.52, 0.52), rim=(1.45, 1.05, 0.62), haze=(0.92, 0.62, 0.66)),
    'sunrise': dict(hi=(1.1, 0.92, 0.66), lit=(1.02, 0.76, 0.5), lit_lo=(0.95, 0.6, 0.58), mid=(0.76, 0.47, 0.6),
                    shade=(0.37, 0.33, 0.52), deep=(0.14, 0.12, 0.32), sky=(0.47, 0.43, 0.62),
                    bounce=(0.86, 0.56, 0.58), rim=(1.45, 1.1, 0.7), haze=(0.94, 0.7, 0.72)),
}
PRESETS['summer'] = PRESETS['noon']
PALETTES = PRESETS


def palette(p):
    if isinstance(p, str):
        p = PRESETS[p]
    return {k: np.asarray(_c(v), F32) for k, v in p.items()}


def mix_palette(a, b, t):
    a, b = palette(a), palette(b)
    return {k: (a[k] * (1 - t) + b[k] * t).astype(F32) for k in a}


# ----------------------------------------------------------------------------------------- small helpers


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _resize(img, w, h, interp=cv2.INTER_LINEAR):
    w, h = max(int(w), 1), max(int(h), 1)
    if img.size == 0:
        return np.zeros((h, w) + img.shape[2:], F32)
    return cv2.resize(img, (w, h), interpolation=interp)


def _blur(img, sigma):
    """Gaussian blur; wide blurs run on a reduced image."""
    if sigma <= 0.3 or img.size == 0:
        return img
    if sigma <= 8.0 or min(img.shape[:2]) < 16:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 4.0
    s = _resize(img, max(int(W / f), 2), max(int(H / f), 2), cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 4.0)
    return _resize(s, W, H, cv2.INTER_LINEAR)


def _noise(w, h, cells, seed, octaves=4, stretch=1.0, angle=0.0):
    """fbm in -1..1 of size (h, w); `cells` = base cells across the width; optional anisotropy."""
    if stretch == 1.0 and angle == 0.0:
        return C.fbm(w, h, cells, octaves, seed=seed) * 2 - 1
    D = int(math.hypot(w, h)) + 4
    # generate on a narrow canvas, then widen it: features are elongated HORIZONTALLY by `stretch`
    nw = int(D / stretch) + 2
    n = C.fbm(nw, D, max(cells / stretch, 1.0), octaves, seed=seed, aspect=True)
    n = _resize(n, D, D)
    M = cv2.getRotationMatrix2D((D / 2, D / 2), angle, 1.0)
    n = cv2.warpAffine(n, M, (D, D), borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (D - h) // 2, (D - w) // 2
    return n[y0:y0 + h, x0:x0 + w] * 2 - 1


def _bleed(rgb, a, sigma=6.0):
    """Fill colour into transparent pixels (so bilinear sampling of straight alpha has no dark fringe)."""
    w = _blur(a, sigma) + 1e-6
    fill = _blur(rgb * a[..., None], sigma) / w[..., None]
    k = np.clip(a * 4, 0, 1)[..., None]
    return (rgb * k + fill * (1 - k)).astype(F32)


def kuwahara(img, r, q=6.0, sharp=None):
    """Generalized Kuwahara filter (painterly flat patches with crisp boundaries). img (h, w, 3) float,
    r = sector size (px). The 4 overlapping square sectors around each pixel are blended with weights
    1 / (1 + var * k) ** q (k normalises by the image's variance) -> soft, stroke-like simplification
    without blocky artefacts."""
    r = int(max(1, round(r)))
    h, w = img.shape[:2]
    P = np.pad(img.astype(np.float64), ((r + 1, r + 1), (r + 1, r + 1), (0, 0)), mode='edge')
    I1 = np.cumsum(np.cumsum(P, 0), 1)
    I2 = np.cumsum(np.cumsum(P * P, 0), 1)
    I1 = np.pad(I1, ((1, 0), (1, 0), (0, 0)))
    I2 = np.pad(I2, ((1, 0), (1, 0), (0, 0)))
    n = float((r + 1) * (r + 1))
    acc = np.zeros((h, w, 3), np.float64)
    wsum = np.zeros((h, w, 1), np.float64)
    k = sharp if sharp is not None else 1.0 / (float(np.var(img)) * 0.02 + 1e-6)
    for (dy, dx) in ((-r, -r), (-r, 0), (0, -r), (0, 0)):
        y0 = r + 1 + dy
        x0 = r + 1 + dx
        ys, xs = slice(y0 + r + 1, y0 + r + 1 + h), slice(x0 + r + 1, x0 + r + 1 + w)
        ys0, xs0 = slice(y0, y0 + h), slice(x0, x0 + w)

        def box(I):
            return I[ys, xs] - I[ys0, xs] - I[ys, xs0] + I[ys0, xs0]
        m = box(I1) / n
        v = (box(I2) / n - m * m).sum(-1, keepdims=True)
        wt = 1.0 / (1.0 + np.maximum(v, 0) * k) ** q
        acc += m * wt
        wsum += wt
    return (acc / wsum).astype(F32)


# ----------------------------------------------------------------------------------------- lobes


class Lobes:
    """Growable list of axis-aligned ellipsoid lobes in CAMERA space (x right, y down, z forward).

    Per lobe: centre c (3), radii r (3), cut_hi / cut_lo = the solid is clipped to
    cut_hi <= dot(P, pn) <= cut_lo (pn = plate's cut normal, (0, 1, 0) = flat horizontal base at
    y = cut_lo and flat top at y = cut_hi), scale = size of the cloud it belongs to (shadow march
    and form-normal blur scale), haze = extra aerial haze 0..1, warm = extra warm tint 0..1."""

    def __init__(self):
        self.rows = []

    def add(self, c, r, cut_hi=-BIG, cut_lo=BIG, scale=1.0, haze=0.0, warm=0.0, root=-1, dark=0.0):
        """root: index (in this Lobes) of the head lobe this floret grew from (-1 = itself); florets are
        shaded with their head's normal field, so a head reads as ONE cauliflower mass."""
        if np.isscalar(r):
            r = (r, r, r)
        i = len(self.rows)
        self.rows.append((c[0], c[1], c[2], r[0], r[1], r[2], cut_hi, cut_lo, scale, haze, warm,
                          i if root < 0 else root, dark))

    def add_grown(self, grown, **kw):
        """Add the output of surface_grow (c, r, local_root) keeping the head links."""
        base = len(self.rows)
        cuts = kw.pop('cut_fn', None)
        for c, r, rt in grown:
            k = dict(kw)
            if cuts is not None:
                k.update(cuts())
            self.add(c, r, root=base + int(rt), **k)

    def extend(self, other):
        base = len(self.rows)
        for row in other.rows:
            self.rows.append(tuple(row[:11]) + (row[11] + base,) + tuple(row[12:13]))

    def array(self):
        return np.asarray(self.rows, np.float64).reshape(-1, 13)

    def __len__(self):
        return len(self.rows)


def grow_cluster(rng, cores, levels=((0.28, 0.55, 2.2), (0.3, 0.55, 1.8), (0.32, 0.6, 1.3)),
                 up=(0.0, -1.0, 0.0), up_bias=1.1, view_bias=0.7, sun=None, sun_bias=0.5, down_cut=0.25,
                 clump=0.7, flat=1.0, min_r=0.6):
    """Fractal floret growth on a lobe cluster.

    cores - list of (c (3,), r (3,)) big lobes. levels - per generation (rmin, rmax, mean children per
    parent) as fractions of the parent radius; the size is drawn from a power law (many small, few big).
    Children sit on the parent surface in a direction biased `up` (and toward the viewer / sun); they
    are skipped when the direction points down more than `down_cut` (bases stay smooth and flat) and
    when the child would be buried inside other lobes. `clump` gives each parent a random fertility, so
    some stretches carry dense cauliflower while others stay broad and smooth. `flat` < 1 squashes the
    florets vertically. Returns a list of (c, r) with the cores first."""
    up = np.asarray(up, np.float64)
    view = np.array([0.0, 0.0, -1.0])
    out = [(np.asarray(c, np.float64), np.asarray(r, np.float64)) for c, r in cores]
    parents = list(out)
    for (rmin, rmax, mean) in levels:
        new = []
        C_ = np.array([c for c, _ in out])
        R_ = np.array([r for _, r in out])
        for (pc, pr) in parents:
            fert = mean * (1 - clump + clump * 2 * rng.random() ** 1.5)
            n = rng.poisson(fert)
            for _ in range(n):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d) + 1e-9
                d = d + up * up_bias + view * view_bias
                if sun is not None:
                    d = d + np.asarray(sun) * sun_bias
                d /= np.linalg.norm(d) + 1e-9
                if np.dot(d, -up) > down_cut:
                    continue
                s = rmin + (rmax - rmin) * rng.random() ** 2.2
                rr = pr.mean() * s
                if rr < min_r:
                    continue
                r3 = np.array([rr * rng.uniform(1.0, 1.25), rr * flat * rng.uniform(0.85, 1.0), rr])
                cc = pc + d * pr * rng.uniform(0.72, 0.92)
                tip = cc + d * r3.mean() * 0.8
                # buried test: the child's outer tip must be outside every other lobe
                q = (tip[None, :] - C_) / R_
                if np.any(np.einsum('ij,ij->i', q, q) < 0.9):
                    continue
                new.append((cc, r3))
        out.extend(new)
        parents = new
        if not new:
            break
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _exposed(P, Cs, Rs, idx, thr):
    """True where point P[i] lies outside every ellipsoid except its parent idx[i]."""
    M = P.shape[0]
    N = Cs.shape[0]
    ok = np.ones(M, np.bool_)
    for i in prange(M):
        for j in range(N):
            if j == idx[i]:
                continue
            a = (P[i, 0] - Cs[j, 0]) / Rs[j, 0]
            b = (P[i, 1] - Cs[j, 1]) / Rs[j, 1]
            c = (P[i, 2] - Cs[j, 2]) / Rs[j, 2]
            if a * a + b * b + c * c < thr:
                ok[i] = False
                break
    return ok


def surface_grow(rng, lobes, sizes, dens=0.35, area=None, up=(0.0, -1.0, 0.0), front=0.45, down_cut=0.25,
                 clump=0.6, clump_freq=None, flat=1.0, sun=None, sun_bias=0.3, sink=(0.2, 0.55), jitter=(0.7, 1.35),
                 region=None):
    """Multi-scale cauliflower growth: for each radius in `sizes` (px, big -> small) sample points on the
    EXPOSED surface of the current lobe union and add a lobe there (centre sunk by `sink` x r).

    dens: coverage (lobes x r^2 / area); area: surface area estimate (px^2); up: world up (camera space);
    front: thinning of lobes on camera-facing surfaces (detail lives on the silhouette and tops);
    down_cut: no growth where the surface faces down more than this (flat, smooth bases);
    clump: 0..1 low-frequency 3D modulation of the density (dense cauliflower passages vs smooth broad
    ones); flat: vertical squash of new lobes; sun / sun_bias: extra growth on the sunward side;
    jitter: radius spread; region: optional callable(P (M, 3)) -> bool mask to restrict growth.
    lobes: list of (c, r3); returns the grown list."""
    up = np.asarray(up, np.float64)
    Cs = np.array([l_[0] for l_ in lobes], np.float64)
    Rs = np.array([l_[1] for l_ in lobes], np.float64)
    Ro = np.array([l_[2] if len(l_) > 2 else i for i, l_ in enumerate(lobes)], np.int64)
    if area is None:
        area = float(np.sum(Rs.mean(1) ** 2) * 2.0)
    span = float(np.ptp(Cs, axis=0).max() + Rs.max() * 2)
    cf = clump_freq if clump_freq is not None else 2.5 / max(span, 1)
    K = rng.normal(size=(6, 3)) * cf * 6.28
    ph = rng.uniform(0, 6.28, 6)
    for rk in sizes:
        Rm = Rs.mean(1)
        wts = Rm ** 2 * (Rm >= rk * 1.25)
        if wts.sum() <= 0:
            continue
        n = int(dens * area / (rk * rk) * (1 + 0.5 * sizes.index(rk)))
        if n <= 0:
            continue
        n = min(n, 40000)
        idx = rng.choice(len(Rm), n, p=wts / wts.sum())
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        P = Cs[idx] + d * Rs[idx]
        # outward normal of the parent ellipsoid at P
        nrm = (P - Cs[idx]) / Rs[idx] ** 2
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        u = nrm @ up
        pr = np.ones(n)
        pr *= (u > -down_cut)
        pr *= 1 - front * np.clip(-nrm[:, 2], 0, 1) ** 1.5
        pr *= np.where(nrm[:, 2] > 0.45, 0.0, 1.0)          # back side: invisible, skip
        if sun is not None:
            pr *= 1 + sun_bias * (nrm @ np.asarray(sun))
        if clump:
            f = np.sin(P @ K.T + ph).mean(1) * 2.0
            pr *= np.clip(1 - clump + clump * (0.5 + f), 0, 1.5)
        if region is not None:
            pr *= region(P)
        keep = rng.random(n) < pr * 0.8
        P, nrm, idx = P[keep], nrm[keep], idx[keep]
        if len(P) == 0:
            continue
        # buried test against every other lobe (chunked)
        ok = _exposed(np.ascontiguousarray(P), np.ascontiguousarray(Cs), np.ascontiguousarray(Rs),
                      np.ascontiguousarray(idx.astype(np.int64)), 0.97)
        P, nrm, idx = P[ok], nrm[ok], idx[ok]
        m = len(P)
        if m == 0:
            continue
        r = rk * rng.uniform(jitter[0], jitter[1], m)
        k = rng.uniform(sink[0], sink[1], m)
        cn = P - nrm * (r * k)[:, None]
        r3 = np.stack([r * rng.uniform(1.0, 1.2, m), r * flat * rng.uniform(0.85, 1.0, m), r], 1)
        Cs = np.concatenate([Cs, cn], 0)
        Rs = np.concatenate([Rs, r3], 0)
        Ro = np.concatenate([Ro, Ro[idx]], 0)
    return [(Cs[i], Rs[i], Ro[i]) for i in range(len(Cs))]


# ----------------------------------------------------------------------------------------- raster


@njit(cache=True)
def _bboxes(E, f, ppx, ppy, W, H):
    N = E.shape[0]
    bb = np.empty((N, 4), np.int64)
    for i in range(N):
        cx, cy, cz = E[i, 0], E[i, 1], E[i, 2]
        rm = max(E[i, 3], max(E[i, 4], E[i, 5]))
        if cz - rm <= 1e-3:
            bb[i, 0] = 0
            bb[i, 1] = 0
            bb[i, 2] = W - 1
            bb[i, 3] = H - 1
            continue
        zz = cz - rm
        u0 = min(f * (cx - rm) / zz, f * (cx - rm) / (cz + rm)) + ppx
        u1 = max(f * (cx + rm) / zz, f * (cx + rm) / (cz + rm)) + ppx
        v0 = min(f * (cy - rm) / zz, f * (cy - rm) / (cz + rm)) + ppy
        v1 = max(f * (cy + rm) / zz, f * (cy + rm) / (cz + rm)) + ppy
        bb[i, 0] = max(int(math.floor(u0)) - 1, 0)
        bb[i, 1] = max(int(math.floor(v0)) - 1, 0)
        bb[i, 2] = min(int(math.ceil(u1)) + 1, W - 1)
        bb[i, 3] = min(int(math.ceil(v1)) + 1, H - 1)
    return bb


@njit(parallel=True, cache=True, fastmath=True)
def _raycast(E, bb, W, H, f, ppx, ppy, pnx, pny, pnz):
    """Analytic ray cast of clipped ellipsoids. Returns t0 (front distance along the unit ray, inf = sky),
    t1 (back of the solid chain), normal (3), sid (front lobe id)."""
    N = E.shape[0]
    t0 = np.full((H, W), np.inf, np.float32)
    t1 = np.zeros((H, W), np.float32)
    nx = np.zeros((H, W), np.float32)
    ny = np.zeros((H, W), np.float32)
    nz = np.zeros((H, W), np.float32)
    sid = np.full((H, W), -1, np.int32)
    band = 16
    nb = (H + band - 1) // band
    for b in prange(nb):
        ya = b * band
        yb = min(H, ya + band)
        for i in range(N):
            if bb[i, 3] < ya or bb[i, 1] >= yb:
                continue
            cx, cy, cz = E[i, 0], E[i, 1], E[i, 2]
            rx, ry, rz = E[i, 3], E[i, 4], E[i, 5]
            chi, clo = E[i, 6], E[i, 7]
            icx, icy, icz = cx / rx, cy / ry, cz / rz
            cc = icx * icx + icy * icy + icz * icz - 1.0
            for y in range(max(ya, bb[i, 1]), min(yb, bb[i, 3] + 1)):
                dy0 = (y + 0.5 - ppy) / f
                for x in range(bb[i, 0], bb[i, 2] + 1):
                    dx0 = (x + 0.5 - ppx) / f
                    inv = 1.0 / math.sqrt(dx0 * dx0 + dy0 * dy0 + 1.0)
                    dx, dy, dz = dx0 * inv, dy0 * inv, inv
                    ax, ay, az = dx / rx, dy / ry, dz / rz
                    a = ax * ax + ay * ay + az * az
                    bh = ax * icx + ay * icy + az * icz
                    disc = bh * bh - a * cc
                    if disc <= 0.0:
                        continue
                    sq = math.sqrt(disc)
                    tin = (bh - sq) / a
                    tout = (bh + sq) / a
                    if tout <= 0.0:
                        continue
                    # clip planes
                    dp = dx * pnx + dy * pny + dz * pnz
                    lo_t = -1e30
                    hi_t = 1e30
                    lo_n = 0  # 1 = entered through the base (cut_lo), -1 = through the top (cut_hi)
                    if dp > 1e-9:
                        lo_t = chi / dp
                        hi_t = clo / dp
                        lo_n = -1
                    elif dp < -1e-9:
                        lo_t = clo / dp
                        hi_t = chi / dp
                        lo_n = 1
                    elif not (chi <= 0.0 <= clo):
                        continue
                    ent = tin
                    plane = 0
                    if lo_t > ent:
                        ent = lo_t
                        plane = lo_n
                    ex = min(tout, hi_t)
                    if ent >= ex or ent <= 0.0:
                        continue
                    cur = t0[y, x]
                    if ent < cur:
                        if ex >= cur:
                            t1[y, x] = max(ex, t1[y, x])
                        else:
                            t1[y, x] = ex
                        t0[y, x] = ent
                        sid[y, x] = i
                        if plane != 0:
                            nx[y, x] = pnx * plane
                            ny[y, x] = pny * plane
                            nz[y, x] = pnz * plane
                        else:
                            px, py, pz = ent * dx - cx, ent * dy - cy, ent * dz - cz
                            gx, gy, gz = px / (rx * rx), py / (ry * ry), pz / (rz * rz)
                            gl = 1.0 / math.sqrt(gx * gx + gy * gy + gz * gz + 1e-30)
                            nx[y, x] = gx * gl
                            ny[y, x] = gy * gl
                            nz[y, x] = gz * gl
                    elif ent <= t1[y, x] and ex > t1[y, x]:
                        t1[y, x] = ex
    return t0, t1, nx, ny, nz, sid


@njit(parallel=True, cache=True, fastmath=True)
def _march(t0, t1, sid, scale, W, H, f, ppx, ppy, lx, ly, lz, nsteps, s0, growth, soft):
    """Optical depth (in units of the lobe's cloud scale) from each front-surface point toward the sun,
    through the solid between the front (t0) and back (t1) depth buffers (screen-space volume march)."""
    od = np.zeros((H, W), np.float32)
    for y in prange(H):
        dy0 = (y + 0.5 - ppy) / f
        for x in range(W):
            T = t0[y, x]
            if not (T < 1e29):
                continue
            sc = scale[sid[y, x]]
            dx0 = (x + 0.5 - ppx) / f
            inv = 1.0 / math.sqrt(dx0 * dx0 + dy0 * dy0 + 1.0)
            Px, Py, Pz = T * dx0 * inv, T * dy0 * inv, T * inv
            ds = s0 * sc
            hsh = ((x * 73856093) ^ (y * 19349663)) % 1024 / 1024.0
            pos = -ds * hsh
            acc = 0.0
            sw = soft * sc
            for k in range(nsteps):
                pos += ds
                Qx, Qy, Qz = Px + lx * pos, Py + ly * pos, Pz + lz * pos
                if Qz <= 1e-3:
                    break
                u = int(f * Qx / Qz + ppx)
                v = int(f * Qy / Qz + ppy)
                if u < 0 or v < 0 or u >= W or v >= H:
                    break
                a = t0[v, u]
                if a < 1e29:
                    dist = math.sqrt(Qx * Qx + Qy * Qy + Qz * Qz)
                    b = t1[v, u]
                    w1 = (dist - a) / sw
                    w2 = (b - dist) / sw
                    if w1 > 0.0 and w2 > 0.0:
                        acc += ds * min(w1, 1.0) * min(w2, 1.0)
                ds *= growth
            od[y, x] = acc / sc
    return od


# ----------------------------------------------------------------------------------------- painter


def _wblur_normals(n, hit, sig):
    w = _blur(hit, sig) + 1e-4
    out = np.stack([_blur(n[..., k] * hit, sig) / w for k in range(3)], -1)
    return out


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def render_lobes(lobes, w, h, f, ppx, ppy, L, pal, ss=2, pn=(0.0, 1.0, 0.0), up=(0.0, -1.0, 0.0),
                 form=0.55, form_px=None, wrap=0.35, density=2.2, march=(20, 0.035, 1.12, 0.06),
                 bands=0.6, band_noise=0.06, terminator=0.35, rim=1.0, rim_back=1.0, glow=0.0,
                 fog=0.0, fog_ref=None, haze=0.0, lost=1.0, lost_px=None, base_soft=1.0, seed=0,
                 mottle=0.02, key=1.0, sky_fill=0.75, bounce=0.5, deep_amt=0.6, lit_curve=1.0, x0=0, y0=0,
                 lobe_px=None, inflate=0.75, inflate_px=None, head=0.6, head_px=None, micro=0.06, paint=1.0, dab_amt=0.025, warp=1.0, contrast=0.45, pivot=0.3, shadow_soft=0.35, edge=1.0, edge_px=None, rim_px=2.2, crest_focus=None, dimple=1.0, limb=0.25, limb_px=None, keep=None,
                 out=None):
    """Paint a lobe cluster into an RGBA plate (h, w) (or into the window (x0, y0) of `out`).

    lobes - Lobes or (N, 11) array (camera space); f, ppx, ppy - pinhole camera in plate px;
    L - unit vector toward the sun (camera space); pal - palette; ss - supersampling;
    pn - clip-plane normal; up - world up in camera space (for sky fill / bounce).
    form - weight of the blurred big-form normal (0 = pure lobe normals / clay, 1 = one smooth blob);
    form_px - blur of the big-form normal in plate px (default from the lobe scale);
    wrap - light wrap; density - shadow density per cloud scale; march - (steps, first step, growth,
    softness) in cloud-scale units; bands - painted posterisation amount 0..1; band_noise - brush
    wobble of the value boundaries; terminator - strength of the saturated terminator band;
    rim / rim_back - lining on thin sun-facing edges / backlit forward-scatter glow; glow - flat
    translucent glow added to the lit side; fog - aerial perspective per unit depth beyond fog_ref;
    haze - flat haze mix; lost - softness of shadow-side silhouettes; base_soft - softness of flat bases;
    mottle - painterly colour mottling; key - overall light multiplier; sky_fill / bounce / deep_amt -
    how much the shadow picks up sky blue / warm bounce / deep core colour; lit_curve > 1 keeps the
    lit face brighter longer."""
    E = lobes.array() if isinstance(lobes, Lobes) else np.asarray(lobes, np.float64)
    pal = palette(pal) if not isinstance(pal, dict) else pal
    W2, H2 = int(round(w * ss)), int(round(h * ss))
    f2, px2, py2 = f * ss, (ppx - x0) * ss, (ppy - y0) * ss
    if len(E) == 0:
        res = np.zeros((h, w, 4), F32)
        return res
    E = E.copy()
    bb = _bboxes(E, f2, px2, py2, W2, H2)
    t0, t1, nx, ny, nz, sid = _raycast(E, bb, W2, H2, f2, px2, py2, pn[0], pn[1], pn[2])
    hit = np.isfinite(t0)
    L = np.asarray(L, np.float64)
    L = L / np.linalg.norm(L)
    scale = E[:, 8].astype(np.float64)
    # the shadow is soft: march on a strided grid (every q-th pixel) and upsample
    q = 2 if min(W2, H2) > 64 else 1
    t0q, t1q, sidq = (np.ascontiguousarray(a_[::q, ::q]) for a_ in (t0, t1, sid))
    Hq, Wq = t0q.shape
    odq = _march(t0q, t1q, sidq, scale, Wq, Hq, f2 / q, px2 / q, py2 / q, L[0], L[1], L[2], int(march[0]),
                 march[1], march[2], march[3])
    od = _resize(odq, W2, H2, cv2.INTER_LINEAR) if q > 1 else odq
    hitf = hit.astype(F32)
    Ns = np.stack([nx, ny, nz], -1)
    sidc = np.where(hit, sid, 0)
    sc_px = np.where(hit, scale[sidc] * f2 / np.where(hit, t0, 1.0), 0).astype(F32)   # cloud scale in px
    if form_px is None:
        med = float(np.median(sc_px[hit])) if hit.any() else 10.0
        fsig = 0.22 * med
    else:
        fsig = form_px * ss
    if lobe_px is None:
        lobe_px = fsig / ss * 0.9
    if head_px is None:
        head_px = 0.2 * lobe_px
    if edge_px is None:
        edge_px = max(1.5, 0.07 * lobe_px)
    Nb = _wblur_normals(Ns, hitf, fsig)
    Nb2 = _wblur_normals(Ns, hitf, fsig * 0.3)
    if inflate:
        # big form = the silhouette inflated into a rounded dome (radius inflate_px), plus the blurred
        # relief of the lobes: one continuous mass with a readable terminator
        R = (inflate_px if inflate_px is not None else 0.45 * np.sqrt(hit.sum() / np.pi) / ss) * ss
        D = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), 1e4).astype(F32)
        D = np.minimum(D, R)
        hgt = np.sqrt(np.maximum(2 * R * D - D * D, 0))
        hgt = _blur(hgt, 0.12 * R)
        gy_, gx_ = np.gradient(hgt)
        Ni = _norm(np.stack([-gx_, -gy_, -np.ones_like(gx_)], -1))
        Nb = _norm(Ni * inflate + Nb * (1 - inflate) + Nb2 * 0.25)
    Er = E[:, 3:6].mean(1).astype(F32)
    rl = (np.where(hit, Er[sidc], 0) * f2 / np.where(hit, t0, 1.0)).astype(F32)
    # three scales of form: the big mass (inflated silhouette), the heads (normals blurred at head size)
    # and the micro florets (raw lobe normals, weighted by lobe size -> soft painterly dabs, not bubbles)
    # head normal: the ellipsoid normal of the head each floret grew from, at this pixel's 3D point
    hid = E[:, 11].astype(np.int64)[sidc]
    xs_ = (np.arange(W2, dtype=F32)[None, :] + 0.5 - px2) / f2
    ys_ = (np.arange(H2, dtype=F32)[:, None] + 0.5 - py2) / f2
    inv_ = 1.0 / np.sqrt(xs_ * xs_ + ys_ * ys_ + 1.0)
    tq = np.where(hit, t0, 0).astype(F32) * inv_
    Pq = np.stack([tq * xs_, tq * ys_, tq], -1)
    Ch = E[:, 0:3].astype(F32)[hid]
    Rh = E[:, 3:6].astype(F32)[hid]
    Nhd = _norm((Pq - Ch) / (Rh * Rh))
    Nh = _norm(_wblur_normals(Nhd, hitf, head_px * ss) * 0.75 + Nhd * 0.25)
    del Pq, Ch, Rh
    wl = (np.clip(rl / max(lobe_px * ss, 1e-3), 0, 1) ** 1.2 * micro).astype(F32)[..., None]
    N = _norm(Nb * form + Nh * head + Ns * wl)
    Lf = L.astype(F32)
    ndl = N @ Lf
    lam = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
    vis_raw = np.exp(-od * density)
    # soft painted shadow masses: the self-shadow is blurred at the lobe scale (no hard circle arcs)
    ssig = shadow_soft * (lobe_px * ss)
    vis = _blur(vis_raw * hitf, ssig) / (_blur(hitf, ssig) + 1e-4) if shadow_soft else vis_raw
    vis = np.clip(vis, 0, 1)
    direct = lam * vis
    if E.shape[1] > 12:
        dk = E[:, 12].astype(F32)[sidc] * hitf
        direct = direct * (1 - dk)
        vis = vis * (1 - dk)
    # crisp lit edges: where the surface steps back behind a lobe on its sunward side (depth jump toward
    # the sun), paint a thin bright band on the front lobe - the painter's crisp lit top edge
    sdir = np.array([L[0], L[1]], F32)
    sdir = sdir / (np.linalg.norm(sdir) + 1e-6)
    if edge:
        gx0, gy0 = np.meshgrid(np.arange(W2, dtype=F32), np.arange(H2, dtype=F32))
        tt = np.where(hit, t0, 1e9).astype(F32)
        rhd = np.where(hit, Er[E[:, 11].astype(np.int64)][sidc], 0) * f2 / np.where(hit, t0, 1.0)
        rlo = np.maximum(rl * 0.4 + rhd * 0.6, 1.0).astype(F32)
        band = np.zeros_like(t0)
        off = edge_px * ss
        ts = cv2.remap(tt, gx0 + sdir[0] * off, gy0 + sdir[1] * off, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=1e9)
        jump = (ts - tt) * f2 / (rlo * np.maximum(tt, 1e-6))       # depth step in lobe radii
        band = _ss(0.25, 0.9, jump)
        band = np.maximum(band, _blur(band, off * 0.35) * 0.9)
        band *= hitf * np.clip(Ns @ L.astype(F32) + 0.6, 0, 1)
        if crest_focus is not None:
            cx_, cw_, floor_ = crest_focus
            xs2 = (np.arange(W2, dtype=F32)[None, :] / ss + x0 - cx_) / cw_
            cn = _noise(W2, H2, max(W2 / (ss * lobe_px * 1.5), 3), seed + 61, 3)
            band = band * np.clip(floor_ + (1 - floor_) * np.exp(-xs2 * xs2) + 0.35 * cn, 0.05, 1.0)
        band_g = band
        if _DEBUG is not None:
            _DEBUG['band'] = band
    if limb:
        # painters keep the brightest white for the sunward silhouette: the face interior settles to
        # a paler lit_lo (limb brightening of a backlit/sidelit volume)
        Dl = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), 1e4).astype(F32)
        lpx = (limb_px if limb_px is not None else 0.8 * lobe_px) * ss
        direct = direct * (1 - limb * (1 - np.exp(-Dl / lpx)))
    if dimple:
        kd = max(3, int(dimple * lobe_px * ss * 0.25) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        closed = cv2.morphologyEx(direct.astype(F32), cv2.MORPH_CLOSE, ker)
        direct = np.where(hit, closed, direct)
    if lit_curve != 1.0:
        direct = 1 - (1 - direct) ** lit_curve
    if contrast:
        # painters simplify to lit vs shadow: sigmoid around `pivot`, keeping a little of the gradient
        sg = 1 / (1 + np.exp(-(direct - pivot) * (4 + 14 * contrast)))
        s0_, s1_ = 1 / (1 + np.exp(pivot * (4 + 14 * contrast))), 1 / (1 + np.exp(-(1 - pivot) * (4 + 14 * contrast)))
        direct = direct * (1 - contrast) + (sg - s0_) / (s1_ - s0_) * contrast
    if edge:
        # crisp lit crest / top edges land in the lit part of the ramp
        direct = np.maximum(direct, np.clip(band_g * edge * (0.5 + 0.5 * vis_raw), 0, 1))
    rng = np.random.default_rng(seed)
    if warp:
        A_ = warp * lobe_px * ss * 0.12
        wx = _noise(W2, H2, max(W2 / (ss * lobe_px * 0.6), 3), seed + 51, 3) * A_
        wy = _noise(W2, H2, max(W2 / (ss * lobe_px * 0.6), 3), seed + 52, 3) * A_
        gx1, gy1 = np.meshgrid(np.arange(W2, dtype=F32), np.arange(H2, dtype=F32))
        dw = cv2.remap(direct.astype(F32), gx1 + wx, gy1 + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        direct = np.where(hit, dw, direct)
    # painted banding with brushy boundaries
    nz1 = _noise(W2, H2, max(W2 / (ss * 45.0), 3), seed + 11, 4)
    dn = np.clip(direct + band_noise * nz1, 0, 1)
    th = (0.1, 0.3, 0.55, 0.8)
    ws = 0.035
    post = sum(_ss(t - ws, t + ws, dn) for t in th) / len(th)
    v = dn * (1 - bands) + post * bands
    v = np.clip(v * key, 0, 1.2)
    if _DEBUG is not None:
        _DEBUG.update(lam=lam, vis=vis_raw, v=v, direct=direct, od=od, hit=hitf, wl=wl[..., 0], nsl=np.clip(Ns @ Lf * 0.5 + 0.5, 0, 1), nbl=np.clip(Nb @ Lf * 0.5 + 0.5, 0, 1))
    # shadow colour: sky fill on up-facing, warm bounce below, deep indigo in the thick core
    upv = N @ np.asarray(up, F32)
    sh = pal['shade'][None, None, :] * np.ones_like(N)
    sh = sh + (pal['sky'] - pal['shade']) * (sky_fill * _ss(-0.2, 0.7, upv))[..., None]
    sh = sh + (pal['bounce'] - sh) * (bounce * _ss(0.0, 0.9, -upv))[..., None]
    core = _ss(0.6, 3.5, od * density) * deep_amt
    if E.shape[1] > 12:
        core = np.maximum(core, dk)
    sh = sh + (pal['deep'] - sh) * core[..., None]
    # lit ramp
    ramp_pos = np.array([0.0, 0.28, 0.52, 0.8, 1.0], F32)
    col = np.empty(N.shape, F32)
    keys = [None, pal['mid'], pal['lit_lo'], pal['lit'], pal['hi']]
    vv = np.clip(v, 0, 1)
    for c in range(3):
        k0 = sh[..., c]
        out_c = k0.copy()
        for j in range(1, 5):
            a = _ss(ramp_pos[j - 1], ramp_pos[j], vv) if False else np.clip((vv - ramp_pos[j - 1]) /
                                                                              (ramp_pos[j] - ramp_pos[j - 1]), 0, 1)
            prev = out_c
            out_c = np.where(vv > ramp_pos[j - 1], prev + (keys[j][c] - prev) * a, prev)
        col[..., c] = out_c
    # terminator: saturated band near the light/shadow boundary
    if terminator:
        tb = np.exp(-((vv - 0.2) / 0.1) ** 2) * terminator
        col = col + (pal['mid'] * 1.05 - col) * tb[..., None] * 0.5
    # lining + backlit glow on thin edges
    z0 = np.zeros(t0.shape, F32)
    dray = np.stack([(np.arange(W2, dtype=F32)[None, :] + 0.5 - px2) / f2 + z0,
                     (np.arange(H2, dtype=F32)[:, None] + 0.5 - py2) / f2 + z0, z0 + 1], -1)
    dray = _norm(dray)
    # lining only on the OUTER silhouette (distance to the sky), never around interior lobes
    Dsil = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), 1e4).astype(F32)
    rpx = max(rim_px * ss, 1.0)
    thin = np.exp(-(Dsil / rpx) ** 1.5) * hitf
    fwd = np.clip(dray @ Lf, 0, 1) ** 6
    facing = np.clip((N @ Lf + 0.15) / 1.15, 0, 1)
    rimv = thin * (rim * facing * np.clip(vis * 1.5, 0, 1) + rim_back * fwd * 2.5)
    col = col + (pal['rim'][None, None, :] - col) * np.clip(rimv * 0.45, 0, 1)[..., None]
    if glow:
        # forward scattering only where the cloud is thin toward the light: silhouettes and crests
        gthin = np.maximum(thin, band_g if edge else 0)
        col = col + pal['hi'] * (glow * fwd * gthin * np.exp(-od * density * 0.3))[..., None]
    # painterly mottling (broad, low amplitude)
    if mottle:
        m1 = _noise(W2, H2, max(W2 / (ss * 160.0), 2), seed + 23, 3)
        col = col * (1 + mottle * m1[..., None])
    # aerial perspective
    hz = np.full(t0.shape, haze, F32)
    if fog:
        ref = fog_ref if fog_ref is not None else float(np.min(t0[hit])) if hit.any() else 1.0
        hz = 1 - (1 - hz) * np.exp(-fog * np.maximum(np.where(hit, t0, ref) - ref, 0))
    hz = 1 - (1 - hz) * (1 - np.clip(E[sidc, 9].astype(F32), 0, 1))
    col = col + (pal['haze'] - col) * (hz * hitf)[..., None]
    warm = (E[sidc, 10].astype(F32)) * hitf
    if np.any(warm > 0):
        col = col * (1 + warm[..., None] * np.array([0.08, 0.0, -0.08], F32))
    # downsample (premultiplied)
    if keep is not None:
        # only pixels whose front lobe is flagged in `keep` (shading still sees every lobe)
        hitf = hitf * np.asarray(keep, F32)[sidc]
    pm = col * hitf[..., None]
    A = _resize(hitf, w, h, cv2.INTER_AREA)
    RGB = _resize(pm, w, h, cv2.INTER_AREA) / np.maximum(A, 1e-6)[..., None]
    RGB = _bleed(np.nan_to_num(RGB).astype(F32), np.clip((A - 0.3) * 2, 0, 1), max(2.0, 0.003 * w))
    if paint:
        # painterly pass: mid-frequency value noise (dabs) + generalized Kuwahara -> flat brush patches
        pr = max(1.0, paint * lobe_px * 0.09)
        dab = _noise(w, h, max(w / (lobe_px * 0.35), 4), seed + 41, 3)
        lumw = np.clip(RGB.mean(-1, keepdims=True), 0, 1)
        RGB = RGB * (1 + dab_amt * dab[..., None] * (0.4 + 0.6 * lumw))
        if pr >= 1.5:
            RGB = kuwahara(RGB, pr, q=4.0)
    # lost edges: soften the silhouette where it faces away from the sun, and along flat bases
    if lost or base_soft:
        sig = (lost_px if lost_px is not None else max(1.5, 0.012 * w))
        Ab = _blur(A, sig)
        gy_, gx_ = np.gradient(_blur(A, sig * 0.7))
        gl = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-6
        onx, ony = -gx_ / gl, -gy_ / gl          # outward silhouette normal (screen)
        Ls = np.array([L[0], L[1]], F32)
        Ls = Ls / (np.linalg.norm(Ls) + 1e-6)
        away = np.clip(-(onx * Ls[0] + ony * Ls[1]), 0, 1)
        down = np.clip(ony, 0, 1) ** 2
        k = np.clip(lost * away ** 1.5 * 0.9 + base_soft * down, 0, 1)
        As = np.minimum(A, _ss(0.2, 0.95, Ab))       # softened INWARD only (no halo in the sky)
        A = A * (1 - k) + As * k
    A = np.clip(A, 0, 1).astype(F32)
    RGB = _bleed(RGB.astype(F32), (A > 0.02).astype(F32) * np.clip(_resize(hitf, w, h, cv2.INTER_AREA) * 3, 0, 1),
                 max(3.0, 0.004 * w))
    res = np.dstack([RGB, A]).astype(F32)
    if out is not None:
        _over_into(out, res, x0, y0)
        return out
    return res


def _over_into(dst, src, x0, y0):
    """Straight-alpha 'over' of src (h, w, 4) into dst at (x0, y0) (dst modified in place)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return dst
    s = src[ya - y0:yb - y0, xa - x0:xb - x0]
    d = dst[ya:yb, xa:xb]
    sa = s[..., 3:4]
    da = d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    # keep bled colour where both are empty
    rgb = np.where(oa > 1e-5, rgb, s[..., :3] * 0.5 + d[..., :3] * 0.5)
    dst[ya:yb, xa:xb, :3] = rgb
    dst[ya:yb, xa:xb, 3:4] = oa
    return dst


# ----------------------------------------------------------------------------------------- generators


def _profile(kind, s, top_round=0.18):
    """Envelope half-width (fraction of the half width) at normalised height s (0 base .. 1 top)."""
    s = np.clip(s, 0, 1)
    if kind == 'cu':           # fair-weather heap: broad dome, slightly narrowing top
        return (0.72 + 0.28 * np.sin(np.pi * np.clip(s * 1.3, 0, 1))) * np.sqrt(np.clip(1 - s ** 2.6, 0, 1))
    if kind == 'tcu':          # towering cumulus: tapering column with a billowing dome top
        return (0.95 - 0.35 * s) * np.sqrt(np.clip(1 - s ** 3.0, 0, 1)) * (0.8 + 0.2 * np.cos(s * 9.0))
    if kind == 'bank':         # low wide heap with shoulders
        return np.sqrt(np.clip(1 - s ** 3.2, 0, 1))
    # 'cb' towering column: broad base, tapering, rounded crown
    w = 0.95 - 0.38 * s
    cap = np.sqrt(np.clip(1 - np.clip((s - (1 - top_round)) / top_round, 0, 1) ** 2, 0, 1))
    return w * (0.35 + 0.65 * cap)


def cloud_lobes(rng, X, Yb, Z, width, height, kind='cu', lean=0.0, wobble=0.08, depth=0.8, detail=1.0,
                sun3=None, base_jitter=0.012, anvil=None, heads=None, scale=None, haze=0.0, warm=0.0,
                cut_top=None, hang=0.0, levels=None, flat=1.0, core_flat=1.0, min_r=0.7):
    """Build the lobe cluster of one cloud (camera space, units = plate px at the cloud's depth).

    X, Yb, Z: base centre (Yb = base height, y down); width / height; kind: 'cu' heap, 'cb' tower,
    'bank' low heap; lean: top offset / height; wobble: sideways meander of the column; depth: front
    to back thickness / width; detail: floret density multiplier; sun3: unit sun vector (florets
    favour the sun side); anvil: None or dict(left, right, thick, top) for a cumulonimbus anvil
    (px extents from the column axis; top = y of the flat top, default top of the column);
    heads: number of surface heads (default by size); hang: fraction of lobes allowed to hang below
    the base (breaks a ruler-straight base). Returns a Lobes."""
    L = Lobes()
    sc = scale if scale is not None else 0.5 * max(width, height * 0.6)
    hw = width * 0.5
    ph = rng.uniform(0, 6.28, 3)

    def axis_x(s):
        return X + lean * height * s + wobble * hw * (np.sin(2.2 * s * np.pi + ph[0]) * 0.6 +
                                                     0.4 * np.sin(4.1 * s * np.pi + ph[1]))

    def base_cut():
        return Yb + base_jitter * height * rng.normal()

    ct = cut_top if cut_top is not None else -BIG
    cores = []
    # interior fill: lobes along the axis, kept inside the envelope (no holes, never seen as balls)
    nfill = max(int(height / (hw * 0.45)) + 2, 3)
    for i in range(nfill):
        s = (i + 0.5) / nfill * 0.95
        pw = float(_profile(kind, s)) * hw
        r = max(pw * rng.uniform(0.5, 0.62), 0.02 * hw)
        cy = Yb - s * height
        if kind not in ('cb', 'tcu'):
            cy = Yb - min(s * height, height - r * 1.1)
        cores.append((np.array([axis_x(s), cy, Z + rng.uniform(-0.1, 0.1) * hw * depth]),
                      np.array([r, r * core_flat, r * depth])))
    # surface heads: power-law sizes, mostly on the visible (front) half, protruding from the envelope
    area = (height / max(hw, 1)) * 2.2 + 3
    nh = heads if heads is not None else int(area * 12 * (0.5 + 0.5 * detail))
    for i in range(nh):
        s = rng.random() ** 0.8
        pw = float(_profile(kind, s)) * hw
        if pw < 0.02 * hw:
            continue
        phi = rng.uniform(-1.5, 1.5)
        rr = hw * (0.08 + 0.28 * rng.random() ** 1.8) * (0.6 + 0.4 * pw / hw)
        k = pw - rr * rng.uniform(0.1, 0.5)
        ex = axis_x(s) + math.sin(phi) * k
        ez = Z - math.cos(phi) * k * depth
        ey = Yb - s * height
        if kind not in ('cb', 'tcu') or s > 0.8:
            ey = Yb - min(s * height, height - rr * 0.6)
        cores.append((np.array([ex, ey, ez]), np.array([rr * rng.uniform(1.0, 1.2), rr * core_flat, rr])))
    sizes = [hw * f_ for f_ in (levels if levels is not None else (0.12, 0.065, 0.034, 0.018))]
    sizes = [r_ for r_ in sizes if r_ >= min_r]
    area = math.pi * hw * height * 1.3
    lobes = surface_grow(rng, cores, sizes, dens=1.2 * detail, area=area, sun=sun3, flat=flat, clump=0.5,
                         front=0.8)
    L.add_grown(lobes, cut_fn=lambda: dict(cut_lo=base_cut()), cut_hi=ct, scale=sc, haze=haze, warm=warm)
    if anvil is not None:
        _anvil(L, rng, axis_x(1.0), anvil.get('top', Yb - height), Z, anvil, sc, depth * hw, haze, warm, sun3)
    return L


def _anvil(L, rng, xc, top, Z, a, sc, zspread, haze, warm, sun3):
    """Flat-topped anvil: a dense sheet of flattened lobes, thickest over the column, tapering smoothly
    into a thin downwind blade (and a short blunt upwind end). Crisp flat top (cut plane), gently
    bumpy underside, small florets along the upper surface only."""
    left, right, th = a['left'], a['right'], a['thick']
    n = int(a.get('n', 90)) * 2
    down_right = right >= left
    cores = []
    for i in range(n):
        u = (i + rng.random()) / n * 2 - 1                 # stratified along the anvil
        x = xc + (u * right if u > 0 else u * left)
        downwind = (u > 0) == down_right
        au = abs(u)
        t = th * (1 - 0.88 * au ** 1.3) if downwind else th * (1 - 0.35 * au ** 2)
        rx = t * rng.uniform(2.0, 3.2)
        ry = t * rng.uniform(0.42, 0.55)
        y = top + ry * 0.95 + (th - t) * 0.12 * rng.uniform(0.5, 1.0)
        z = Z + rng.uniform(-1, 1) * zspread * (0.7 - 0.4 * au)
        cores.append((np.array([x, y, z]), np.array([rx, ry, rx * 0.8])))
    lob = surface_grow(rng, cores, [th * 0.22, th * 0.11], dens=0.5, area=(left + right) * th * 3,
                       front=0.5, down_cut=0.2, flat=0.6, sun=sun3, clump=0.7)
    L.add_grown(lob, cut_hi=top, cut_lo=BIG, scale=sc, haze=haze, warm=warm)


# ----------------------------------------------------------------------------------------- sun helpers


def _sun_vec(sun, sun_dir, sun_z, at):
    if sun is not None:
        d = np.array([sun[0] - at[0], sun[1] - at[1]], np.float64)
    else:
        d = np.asarray(sun_dir if sun_dir is not None else (0.6, -0.7), np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    v = np.array([d[0], d[1], -sun_z], np.float64)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------------------- plates


def dissolve_base(plate, cx, half_w, base_y, fade_h, seed=0, amount=0.85, tear=0.6):
    """Break up a flat cloud base in place: alpha fades over `fade_h` px above base_y with a torn,
    horizontally stretched noise edge (wispy, lost base instead of a ruler-straight slab), limited to
    x in cx +- half_w."""
    h, w = plate.shape[:2]
    y0 = int(max(base_y - fade_h * 1.6, 0))
    y1 = int(min(base_y + fade_h * 0.5, h))
    x0 = int(max(cx - half_w, 0))
    x1 = int(min(cx + half_w, w))
    if y1 <= y0 or x1 <= x0:
        return plate
    hh, ww = y1 - y0, x1 - x0
    n = _noise(ww, hh, max(ww / (fade_h * 1.2), 2), seed + 7, 4, stretch=5.0)
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    edge = base_y - fade_h * (0.5 + tear * 0.5 * n)
    f = _ss(edge - fade_h * 0.35, edge + fade_h * 0.6, ys)
    xs = (np.arange(x0, x1, dtype=F32)[None, :] - cx) / max(half_w, 1)
    f = f * np.clip(1.4 - np.abs(xs) * 0.6, 0, 1)
    plate[y0:y1, x0:x1, 3] *= (1 - amount * f)
    return plate


def _side_camera(w, h, eye_y, dist):
    return dict(f=dist, ppx=w / 2.0, ppy=eye_y, dist=dist)


def _render_side(L, w, h, cam, Lsun, pal, ss, **kw):
    """Render lobes of side-view clouds into a full plate, cropping to their screen bbox."""
    E = L.array()
    out = np.zeros((h, w, 4), F32)
    if len(E) == 0:
        return out
    bb = _bboxes(E, cam['f'], cam['ppx'], cam['ppy'], w, h)
    pad = int(0.03 * w) + 8
    x0, y0 = max(int(bb[:, 0].min()) - pad, 0), max(int(bb[:, 1].min()) - pad, 0)
    x1, y1 = min(int(bb[:, 2].max()) + pad, w), min(int(bb[:, 3].max()) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return out
    res = render_lobes(E, x1 - x0, y1 - y0, cam['f'], cam['ppx'], cam['ppy'], Lsun, pal, ss=ss, x0=x0, y0=y0, **kw)
    out[y0:y1, x0:x1] = res
    out[..., :3] = _bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(4.0, 0.01 * w))
    return out


def cumulonimbus_plate(w, h, cx, base_y, height, sun=None, sun_dir=None, preset='noon', seed=0, sun_z=0.15,
                       width=None, anvil=True, anvil_dir=1, anvil_len=0.55, lean=0.0, haze=0.0, detail=1.0,
                       ss=1.5, eye_y=None, persp=4.0, kind='tcu', turrets=3, base_fade=0.06,
                       **kw):
    """Towering cumulus / cumulonimbus as an RGBA plate.

    cx, base_y: base centre (plate px); height: base to top (px); width: column width (default
    0.62 * height); anvil: flat-topped anvil at the top (anvil_dir +1 = spreads right); anvil_len:
    downwind extent / height; lean: top offset / height; haze: aerial haze 0..1; detail: floret
    density; eye_y: horizon (default base_y + 0.05 h: we look very slightly up at the base);
    persp: camera distance / height (smaller = stronger perspective); kind 'cb' (tower) | 'cu' heap.
    Extra keywords go to render_lobes (form, wrap, density, bands, rim, lost, ...)."""
    rng = np.random.default_rng(seed)
    width = width if width is not None else 0.62 * height
    eye = eye_y if eye_y is not None else base_y + 0.05 * h
    dist = persp * height
    cam = _side_camera(w, h, eye, dist)
    X, Yb = cx - cam['ppx'], base_y - eye
    Ls = _sun_vec(sun, sun_dir, sun_z, (cx, base_y - height * 0.6))
    an = None
    if anvil:
        L_ = anvil_len * height
        an = dict(left=L_ * 0.3 if anvil_dir > 0 else L_, right=L_ if anvil_dir > 0 else L_ * 0.3,
                  thick=0.07 * height, n=int(90 * detail))
    lob = cloud_lobes(rng, X, Yb, dist, width, height * (0.9 if anvil else 1.0), kind=kind, lean=lean,
                      detail=detail, sun3=Ls, anvil=an, haze=haze, scale=0.5 * width)
    # secondary turrets: lower towers leaning against the main column -> stepped, grouped massing
    for k in range(int(turrets)):
        side = (-1) ** k * (1 if rng.random() < 0.5 else -1) if k else float(rng.choice([-1, 1]))
        th_ = height * rng.uniform(0.35, 0.7) * (0.85 ** k)
        tw_ = min(width * rng.uniform(0.55, 0.85), th_ * 0.85)
        off = side * (width * 0.5 + tw_ * rng.uniform(0.05, 0.3))
        dz = rng.uniform(-0.25, 0.35) * width
        sub = cloud_lobes(np.random.default_rng(seed * 31 + k + 1), X + off, Yb + rng.uniform(0, 0.02) * height,
                          dist + dz, tw_, th_, kind='tcu', lean=-side * rng.uniform(0.0, 0.12), detail=detail,
                          sun3=Ls, haze=haze, scale=0.5 * width)
        lob.extend(sub)
    pal = palette(preset)
    kw.setdefault('form_px', 0.09 * width)
    out = _render_side(lob, w, h, cam, Ls, pal, ss, seed=seed, **kw)
    if base_fade:
        dissolve_base(out, cx, width * 1.6, base_y, base_fade * height, seed=seed)
    return out


def cumulus_plate(w, h, clouds, sun=None, sun_dir=None, preset='noon', sun_z=0.15, detail=1.0, ss=1.5,
                  eye_y=None, persp=5.0, base_fade=0.12, **kw):
    """Several fair-weather cumulus heaps on one plate, rendered in ONE pass (shared camera, z-buffered).
    clouds = [(cx, base_y, width, height, seed[, haze]), ...] in plate px; later entries are nearer."""
    if not clouds:
        return np.zeros((h, w, 4), F32)
    pal = palette(preset)
    big = max(max(c[2], c[3]) for c in clouds)
    eye = eye_y if eye_y is not None else max(c[1] for c in clouds) + 0.05 * h
    D = persp * big
    cam = _side_camera(w, h, eye, D)
    L = Lobes()
    xs = [c[0] for c in clouds]
    Ls = _sun_vec(sun, sun_dir, sun_z, (float(np.mean(xs)), float(np.mean([c[1] - c[3] * 0.5 for c in clouds]))))
    n = len(clouds)
    for i, cl in enumerate(clouds):
        cx, by, cw, ch, sd = cl[:5]
        hz = cl[5] if len(cl) > 5 else 0.0
        z = D * (1.0 + 0.25 * (n - 1 - i) / max(n, 1))       # earlier entries sit further back
        k = z / D
        rng = np.random.default_rng(sd)
        lob = cloud_lobes(rng, (cx - cam['ppx']) * k, (by - eye) * k, z, cw * k, ch * k, kind='cu', detail=detail,
                          sun3=Ls, haze=hz, depth=0.7, min_r=0.6 * k, scale=0.5 * max(cw, ch * 0.6) * k)
        L.extend(lob)
    kw.setdefault('form_px', 0.1 * float(np.median([c[2] for c in clouds])))
    kw.setdefault('inflate_px', 0.35 * float(np.median([c[3] for c in clouds])))
    out = _render_side(L, w, h, cam, Ls, pal, ss, seed=int(clouds[0][4]), **kw)
    if base_fade:
        for cl in clouds:
            dissolve_base(out, cl[0], cl[2] * 0.75, cl[1], base_fade * cl[3], seed=int(cl[4]))
    return out


def horizon_bank_plate(w, h, y, height, sun=None, sun_dir=None, preset='noon', seed=0, rows=3, haze=0.35,
                       sun_z=0.2, ss=1.5, detail=0.8, **kw):
    """Low bank of heaped cumulus along the horizon (base at y), `rows` receding rows (back rows lower,
    smaller, hazier)."""
    rng = np.random.default_rng(seed)
    clouds = []
    for r in range(rows):
        k = (rows - 1 - r) / max(rows - 1, 1)       # 1 = back row
        hh = height * (1 - 0.45 * k)
        x = -0.05 * w + rng.uniform(0, 0.1) * w
        while x < w * 1.05:
            cw = hh * rng.uniform(1.6, 3.4)
            ch = hh * rng.uniform(0.55, 1.0)
            clouds.append((x + cw / 2, y - k * height * 0.08, cw, ch, int(rng.integers(1 << 30)),
                           min(haze + 0.3 * k, 0.9)))
            x += cw * rng.uniform(0.55, 0.85)
    kw.setdefault('lost', 0.6)
    return cumulus_plate(w, h, clouds, sun=sun, sun_dir=sun_dir, preset=preset, sun_z=sun_z, detail=detail, ss=ss,
                         eye_y=y + 0.02 * h, persp=8.0, **kw)


def cirrus_plate(w, h, preset='noon', seed=0, region=(0.05, 0.4), angle=-8.0, density=0.5, opacity=0.6, **kw):
    """High cirrus (hooked mare's tails) tinted by the preset (wraps lib.sky.cirrus_plate)."""
    p = palette(preset)
    col = np.clip(p['lit'] * 0.7 + np.clip(p['hi'], 0, 1) * 0.3, 0, 1)
    return _S.cirrus_plate(w, h, seed=seed, color=tuple(col), under=tuple(np.clip(p['lit_lo'], 0, 1)), angle=angle,
                           density=density, region=region, opacity=opacity, **kw)


def cloudlets_plate(w, h, preset='noon', seed=0, region=(0.05, 0.45), density=0.5, size=0.012, angle=-12.0,
                    stretch=2.2, opacity=0.85, sun_dir=(0.6, -0.8), patch=0.35):
    """Mackerel sky / altocumulus: fields of small broken cloudlets (gow_02) as an RGBA plate.

    region: (y0, y1) band (fractions of h); density: coverage 0..1; size: flake size (fraction of w);
    angle / stretch: flake orientation and elongation; patch: size of the large patches the field
    breaks into (fraction of w); sun_dir: screen direction to the sun (flakes get a lit edge on that
    side and a soft cooler body)."""
    pal = palette(preset)
    rng = np.random.default_rng(seed)
    cells = 1.0 / max(size, 1e-3)
    n = _noise(w, h, cells, seed + 3, 3, stretch=stretch, angle=angle) * 0.5 + 0.5
    big = _noise(w, h, 2.0 / max(patch, 1e-3), seed + 5, 3, stretch=2.5, angle=angle)
    lo_, hi_ = np.percentile(big, 10), np.percentile(big, 90)
    big = np.clip((big - lo_) / (hi_ - lo_ + 1e-6), 0, 1)
    ys = np.arange(h, dtype=F32)[:, None] / h
    band = _ss(region[0], region[0] + 0.08, ys) * (1 - _ss(region[1] - 0.12, region[1], ys))
    field = _ss(0.8 - 0.3 * density, 1.0, big) * band
    th = 0.92 - 0.36 * field * (0.6 + 0.4 * density)
    m = _ss(th, th + 0.07, n) * (field > 0.01)
    # lit edge on the sun side: the flake minus itself shifted toward the sun
    d = np.asarray(sun_dir, np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    k = max(1.0, size * w * 0.18)
    sh = _shift_img(m, -d[0] * k, -d[1] * k)
    rim = _blur(np.clip(m - sh, 0, 1), 0.6)
    body = pal['sky'] * 0.55 + pal['lit_lo'] * 0.45
    col = body[None, None, :] + (np.clip(pal['lit'], 0, 1) - body)[None, None, :] * np.clip(rim * 0.8 + 0.35 * n, 0, 1)[..., None]
    a = np.clip(_blur(m, 0.7) * opacity * (0.45 + 0.55 * np.clip(rim * 2 + 0.3, 0, 1)), 0, 1)
    out = np.dstack([col, a]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (a > 0.02).astype(F32), 3.0)
    return out


def _shift_img(m, dx, dy):
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(m.astype(F32), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def haze_plate(w, h, y0, y1, color, amount=0.6, y2=None):
    """Horizontal aerial-haze wash (RGBA): alpha 0 at y0 -> amount at y1 (-> 0 at y2 if given)."""
    ys = np.arange(h, dtype=F32)
    a = _ss(y0, y1, ys) * amount
    if y2 is not None:
        a = a * (1 - _ss(y1, y2, ys))
    out = np.zeros((h, w, 4), F32)
    out[..., :3] = _c(color)
    out[..., 3] = a[:, None]
    return out


# ----------------------------------------------------------------------------------------- sea of clouds


def _pitch_cam(w, h, horizon_y, fov):
    """Pinhole camera pitched down so the horizon lands on horizon_y. Returns f, ppx, ppy, theta."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    ppx, ppy = w / 2.0, h / 2.0
    theta = math.atan((ppy - horizon_y) / f)          # > 0: looking down
    return f, ppx, ppy, theta


def _to_cam(P, theta):
    """World (X, Yd (down), Z (forward, horizontal)) -> camera coords for a camera pitched down theta."""
    c, s = math.cos(theta), math.sin(theta)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    return np.stack([X, Y * c - Z * s, Y * s + Z * c], -1)


def _value_noise2(rng, n=7, freq=1.0):
    """Smooth random 2D field (sum of random plane waves) -> callable(X, Z) in ~[-1, 1]."""
    K = rng.normal(size=(n, 2)) * freq
    ph = rng.uniform(0, 6.28, n)
    amp = 1.0 / (1 + np.arange(n) * 0.35)

    def fn(X, Z):
        v = 0.0
        for i in range(n):
            v = v + amp[i] * np.sin(X * K[i, 0] + Z * K[i, 1] + ph[i])
        return v / amp.sum() * 2.2
    return fn


def sea_of_clouds_plate(w, h, horizon_y, sun=None, preset='sunrise', seed=0, towers=(), fov=55.0, cam_height=1.0,
                        near_r=0.06, grow_exp=0.72, far_dist=45.0, valley=0.55, valley_depth=2.2, relief=0.35,
                        detail=1.0, sun_z=0.0, ss=1.25, fog=0.008, horizon_haze=None, haze_color=None,
                        density=1.6, glow=0.5, far_band=True, lift=0.05, floor_dark=0.85, valley_dark=0.8,
                        crest_width=0.3, split=None, **kw):
    """Sea of clouds seen from above (camera above the cloud deck looking toward the horizon), RGBA plate
    (h, w), opaque below the horizon.

    horizon_y: screen row of the horizon; sun: (x, y) screen position of the sun (a physical light at
    infinity: low sun ahead = backlit crests, sunward tops lit gold, deep shaded valleys);
    towers: [dict(x=screen x, dist=world distance (cam_height units), height, width, seed, lean,
    anvil=False, kind='cb'), ...] towering cumulus rising out of the deck; fov: horizontal field of view;
    near_r: head radius at the bottom of the frame (fraction of w); grow_exp: how fast world lobe size
    grows with distance (painter's simplification of far detail: 0 = physical, 1 = constant screen
    size); far_dist: distance (cam_height units) where lobes stop and the painted far band takes over;
    valley: fraction of the deck broken by valleys; valley_depth: their depth (in head radii);
    relief: height variation of the deck (head radii); sun_z: tilts the light toward the viewer (+) to
    light more of the tops; fog: aerial perspective per distance unit; horizon_haze: (colour, amount)
    of the far band; density: shadow density; glow: forward-scatter glow on backlit edges;
    extra keywords go to render_lobes."""
    rng = np.random.default_rng(seed)
    pal = palette(preset)
    f, ppx, ppy, th = _pitch_cam(w, h, horizon_y, fov)
    hc = cam_height
    # distance at the bottom of the frame
    ang_b = math.atan((h - ppy) / f) + th
    d_near = hc / math.tan(max(ang_b, 1e-3))
    R0 = near_r * w * d_near / f          # world head radius at the bottom of frame

    def Rw(d):
        return R0 * np.maximum(d / d_near, 1.0) ** grow_exp

    hfield = _value_noise2(rng, 8, 1.0 / (R0 * 5.0))
    vfield = _value_noise2(rng, 6, 1.0 / (R0 * 9.0))
    L = Lobes()
    # heads on a jittered grid in (X, Z); row spacing follows the local radius
    d = d_near * 0.75
    rows = []
    while d < far_dist * hc:
        R = float(Rw(d))
        rows.append((d, R))
        d += R * 1.25
    band_heads = {}
    for (d, R) in rows:
        halfw = d * (w / 2.0 + 0.12 * w) / f + R
        x = -halfw + rng.uniform(0, R)
        while x < halfw:
            X = x + rng.uniform(-0.3, 0.3) * R
            Z = d + rng.uniform(-0.45, 0.45) * R
            Rr = R * rng.uniform(0.75, 1.3)
            hv = float(hfield(X, Z))
            vv = float(vfield(X, Z))
            top = hc - relief * R * hv
            if vv < -1 + 2 * valley * 0.5:
                k = (-1 + valley - vv) / max(valley, 1e-3)
                top += valley_depth * R * min(k * 1.5, 1.0)
                if rng.random() < min(k * 1.2, 0.85):
                    x += R * rng.uniform(1.0, 1.5)
                    continue
            c = np.array([X, top + Rr * 0.35, Z])
            r3 = np.array([Rr * rng.uniform(1.1, 1.4), Rr * rng.uniform(0.6, 0.8), Rr * rng.uniform(1.0, 1.25)])
            bi = int(math.log(max(d / d_near, 1.0)) / math.log(1.6))
            band_heads.setdefault(bi, []).append((c, r3))
            # skirt: the head's mass continues down into the deck (no floating saucers over valleys)
            band_heads[bi].append((c + np.array([0, Rr * 0.9, Rr * 0.2]), r3 * np.array([1.15, 1.0, 1.1])))
            x += Rr * rng.uniform(1.05, 1.6)
    floor = []
    near_flag = []
    # valley floor: a lower, sparser layer of broad flat lobes (seen through the gaps, deep in shadow)
    for (d, R) in rows[::2]:
        halfw = d * (w / 2.0 + 0.12 * w) / f + R * 2
        x = -halfw
        while x < halfw:
            Rr = R * rng.uniform(1.6, 2.4)
            c = np.array([x, hc + valley_depth * R * 1.3 + Rr * 0.2, d + rng.uniform(-0.5, 0.5) * R])
            floor.append((c, np.array([Rr * 1.4, Rr * 0.4, Rr * 1.2]), float(Rw(d))))
            x += Rr * rng.uniform(1.4, 2.0)
    # florets per distance band (sizes relative to the band's head size)
    up_w = (0.0, -1.0, 0.0)
    for bi in sorted(band_heads):
        heads = band_heads[bi]
        Rb = float(np.median([r[0] for _, r in heads]))
        sizes = [Rb * k for k in (0.42, 0.22, 0.11)]
        sizes = [s_ for s_ in sizes if s_ * f / (d_near * 1.6 ** bi) > 0.9]
        area = float(np.sum([r[0] * r[2] for _, r in heads])) * 2.0
        if sizes:
            grown = surface_grow(rng, heads, sizes, dens=0.9 * detail, area=area, up=up_w, front=0.0,
                                 down_cut=0.1, flat=0.7, clump=0.6)
        else:
            grown = [(c, r, i) for i, (c, r) in enumerate(heads)]
        # convert to camera space
        base = len(L)
        for c, r, rt in grown:
            dist = float(c[2])
            cc = _to_cam(np.asarray(c, np.float64), th)
            Rd = float(Rw(dist))
            dk = float(np.clip((c[1] - hc - 0.5 * Rd) / (valley_depth * Rd), 0, 1)) * valley_dark
            L.add(cc, r, scale=Rd * 3.0, root=base + int(rt), dark=dk)
            near_flag.append(split is not None and dist < split * hc)
    for c, r, Rd in floor:
        L.add(_to_cam(np.asarray(c, np.float64), th), r, scale=Rd * 3.0, dark=floor_dark)
        near_flag.append(split is not None and c[2] < split * hc)
    # towers
    for tw in towers:
        trng = np.random.default_rng(tw.get('seed', 0))
        dist = tw['dist'] * hc
        X = (tw['x'] - ppx) / f * dist
        # the deck top at that distance, projected: we stand the tower on the deck (base sunk in it)
        Hh = tw['height'] * hc
        Wd = tw.get('width', 0.55) * Hh
        an = None
        if tw.get('anvil'):
            an = dict(left=0.2 * Hh, right=0.6 * Hh, thick=0.08 * Hh, n=50)
        tl = cloud_lobes(trng, X, hc + 0.3 * float(Rw(dist)), dist, Wd, Hh, kind=tw.get('kind', 'tcu'),
                         lean=tw.get('lean', 0.0), detail=tw.get('detail', 1.0), depth=0.9, anvil=an,
                         scale=0.5 * Wd, min_r=Wd * 0.004)
        E = tl.array()
        base = len(L)
        for row in E:
            cc = _to_cam(row[0:3], th)
            L.add(cc, row[3:6], scale=row[8], root=base + int(row[11]), warm=tw.get('warm', 0.0))
            near_flag.append(split is not None and dist < split * hc)
    # light: physical sun direction from its screen position, tilted toward the viewer by sun_z
    if sun is None:
        sun = (w * 0.5, horizon_y - 0.05 * h)
    Ls = np.array([(sun[0] - ppx) / f, (sun[1] - ppy) / f, 1.0])
    Ls = Ls / np.linalg.norm(Ls)
    Ls[2] -= sun_z
    Ls = Ls / np.linalg.norm(Ls)
    up_c = np.array([0.0, -math.cos(th), -math.sin(th)])
    Ls = Ls + up_c * lift                       # painter's cheat: tops catch the low sun
    Ls = Ls / np.linalg.norm(Ls)
    kw.setdefault('inflate', 0.0)
    kw.setdefault('form', 0.25)
    kw.setdefault('head', 1.0)
    lp = near_r * w * 0.9
    kw.setdefault('lobe_px', lp)
    kw.setdefault('form_px', lp * 0.8)
    kw.setdefault('lost', 0.0)
    kw.setdefault('base_soft', 0.0)
    kw.setdefault('fog_ref', d_near)
    kw.setdefault('rim_px', 1.6)
    kw.setdefault('contrast', 0.5)
    kw.setdefault('pivot', 0.38)
    kw.setdefault('paint', 0.1)
    kw.setdefault('limb', 0.0)
    kw.setdefault('crest_focus', (sun[0], crest_width * w, 0.3))      # the deck's detail scale varies with distance: keep the pass light
    rk = dict(ss=ss, pn=(0.0, 1.0, 0.0), up=tuple(up_c), density=density, fog=fog, glow=glow, seed=seed)
    near_plate = None
    if split is not None and any(near_flag):
        nf = np.asarray(near_flag, bool)
        E = L.array()
        # near deck: render only the window its lobes cover (+ margin above for the shadows cast on it)
        bbn = _bboxes(E[nf], f, ppx, ppy, w, h)
        yn0 = int(max(bbn[:, 1].min() - 0.12 * h, 0))
        near_plate = np.zeros((h, w, 4), F32)
        near_plate[yn0:] = render_lobes(E, w, h - yn0, f, ppx, ppy, Ls, pal, keep=nf, y0=yn0, **rk, **kw)
        near_plate[..., :3] = _bleed(near_plate[..., :3], (near_plate[..., 3] > 0.02).astype(F32),
                                     max(4.0, 0.01 * w))
        # far plate: only the far lobes (what the near deck uncovers as it drifts)
        idx = np.nonzero(~nf)[0]
        remap = -np.ones(len(E), np.int64)
        remap[idx] = np.arange(len(idx))
        Ef = E[idx].copy()
        rt = remap[Ef[:, 11].astype(np.int64)]
        Ef[:, 11] = np.where(rt >= 0, rt, np.arange(len(idx)))
        res = render_lobes(Ef, w, h, f, ppx, ppy, Ls, pal, **rk, **kw)
    else:
        res = render_lobes(L, w, h, f, ppx, ppy, Ls, pal, **rk, **kw)
    # painted far band (beyond the lobes) + fill: opaque from the horizon down
    ys = np.arange(h, dtype=F32)[:, None]
    hz_col = _c(haze_color) if haze_color is not None else pal['haze']
    deep = pal['deep'] * 0.6 + pal['shade'] * 0.4
    y_far = ppy + f * math.tan(math.atan(hc / (far_dist * hc)) - th)
    t = np.clip((ys - horizon_y) / max(y_far - horizon_y, 1), 0, 1)
    band = np.broadcast_to(hz_col * (1 - 0.35 * t[..., None]) + deep * 0.35 * t[..., None], (h, w, 3)).copy()
    if far_band:
        # thin horizontal strata of far cloud tops catching light
        n1 = _noise(w, h, 3.0, seed + 91, 3, stretch=18.0)
        strata = np.clip(n1 * 0.5 + 0.5, 0, 1) ** 2 * (1 - t) * 0.35
        band = band + (pal['lit'] - band) * strata[..., None]
    fillA = (ys >= horizon_y - 0.5).astype(F32) * np.clip(ys - horizon_y + 1.0, 0, 1)
    fillA = np.broadcast_to(fillA, (h, w)).astype(F32)
    A = res[..., 3:4]
    rgb = res[..., :3] * A + band * (1 - A)
    a = np.maximum(A[..., 0], fillA)
    if horizon_haze is not None:
        hcol, hamt = horizon_haze
        g = np.exp(-((ys - horizon_y) / (0.05 * h)) ** 2) * hamt
        rgb = rgb + (_c(hcol) - rgb) * g[..., None]
    out = np.dstack([rgb, a]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (a > 0.02).astype(F32), max(4.0, 0.01 * w))
    if near_plate is not None:
        return [out, near_plate]
    return out


# ----------------------------------------------------------------------------------------- drift helpers


def drift(plate, W, H, t, speed=(0.003, 0.0), cam=(0.0, 0.0), zoom=1.0, depth=1.0, **kw):
    """Sample a plate into the W x H frame with wind drift + camera parallax (+ optional billow churn),
    see lib.sky.drift. Safe on empty plates."""
    if plate is None or plate.size == 0:
        return np.zeros((H, W, 4), F32)
    return _S.drift(plate, W, H, t, speed=speed, cam=cam, zoom=zoom, depth=depth, **kw)


def screen_pos(p, W, H, plate_size=None, cam=(0.0, 0.0), zoom=1.0, depth=1.0, t=0.0, speed=(0.0, 0.0)):
    """Screen position of plate point p for a plate sampled with drift(..., cam, zoom, depth, t, speed)."""
    if not isinstance(speed, (tuple, list, np.ndarray)):
        speed = (float(speed), 0.0)
    pw, ph = plate_size if plate_size is not None else (W, H)
    z = 1 + (zoom - 1) * depth
    cx = pw / 2 - speed[0] * W * t + cam[0] * depth
    cy = ph / 2 - speed[1] * W * t + cam[1] * depth
    return (W / 2 + (p[0] - cx) * z, H / 2 + (p[1] - cy) * z)


def occluder(*layers):
    """Combined alpha (H, W) of sampled RGBA layers."""
    a = None
    for L in layers:
        la = np.clip(L[..., 3], 0, 1)
        a = la if a is None else a + la * (1 - a)
    return np.clip(a, 0.0, 1.0)


def composite(img, *layers):
    """Straight-alpha 'over' of sampled RGBA layers (far -> near) onto an RGB image."""
    for L in layers:
        img = _F.over_rgba(img, L)
    return img
