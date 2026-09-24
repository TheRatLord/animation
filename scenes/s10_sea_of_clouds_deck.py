"""s10 round 16 - the sea of clouds as a true 3D deck of painted billboards.

World: x right, h up (0 = top of the fog deck the masses rise out of), Z forward (toward the sun).
A pinhole camera (focal F px, horizon row HY, no pitch) at height hc looks down the deck; every row of
cloud masses lives at its own depth Z, so push-in (camera Z), crane-up (hc) and truck (camera x) give
exact perspective parallax: near rows grow and sink ~3x faster than the mid bank, the towers move at mid
speed, the far rows and the sky barely move.

Each row is pre-painted once into a plate (straight-alpha RGBA) in the frame coordinates of the END
camera (its largest on-screen size) and placed per frame with one zoom about the vanishing point plus a
translation (see s10_sea_of_clouds_r11.place_fast).

Every mass is painted the way a background painter blocks one in (not rendered):
  * silhouette: lib.clouds2 cauliflower (a few big heads, florets only on the upper EDGE);
  * value planes: a light term from the BIG form (heavily blurred silhouette normal . light direction,
    plus height in the mass) split into a warm lit top plane (firm, slightly scalloped terminator), a
    soft rose/lavender mid plane and a cool lavender-grey underside (reflected light low down);
  * torn base: the lower third dissolves into the valley haze through a streaky, wind-combed noise
    (colour first melts into the fog, then alpha tears away) - no bowl bottoms;
  * rim: a continuous 2-4 px hot-gold line only where the silhouette faces the sun, on rows near the
    sun axis; thinner and silver-pink further out, none on edges turned away or on far rows;
  * aerial perspective per row: far rows shrink, lose contrast and melt into the pale peach horizon.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, clouds2 as K  # noqa: E402


def _c(h):
    return np.asarray(K._c(h), np.float32)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------------------------------ palette
HAZE_SUN = _c('#fbe0c0')     # horizon haze on the sun axis
HAZE_OFF = _c('#ecd2d4')     # horizon haze away from it (pale rose)
NEAR = dict(hi=_c('#fff4dc'), lit=_c('#ffdcae'), lit_off=_c('#f6c9c0'), mid=_c('#d6a2bf'), mid_off=_c('#b9a0cf'),
            shade=_c('#6f69b6'), shade_off=_c('#6570b2'), refl=_c('#9794cf'), fog=_c('#a8a2d6'),
            fog_sun=_c('#e9cfd8'))
RIM_GOLD = np.array([1.85, 1.58, 1.08], np.float32)
RIM_SILVER = np.array([1.22, 1.12, 1.14], np.float32)


def row_palette(a):
    """Palette of a row at normalised distance a (0 near .. 1 horizon): colours converge on the haze."""
    k = float(np.clip(a, 0, 1)) ** 1.35
    out = {}
    for key, v in NEAR.items():
        tgt = HAZE_SUN if key in ('hi', 'lit', 'mid', 'fog_sun') else HAZE_OFF
        kk = k * (0.92 if key in ('shade', 'shade_off', 'refl') else 1.0)
        out[key] = _lerp(v, tgt, kk).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------ painter
class Plate:
    """Premultiplied accumulation canvas in frame(ref) coordinates offset by (ox, oy)."""

    def __init__(self, ox, oy, w, h, seed, u):
        self.ox, self.oy, self.w, self.h, self.u = int(ox), int(oy), int(w), int(h), u
        self.acc = np.zeros((self.h, self.w, 4), np.float32)
        q = 4
        wq, hq = max(self.w // q, 8), max(self.h // q, 8)
        # wind-combed tear noise (strongly horizontal) + a soft brush noise, plate sized
        n1 = C.fbm(wq, max(hq // 3, 4), wq / (60.0 * u), 4, seed=seed)
        self.tear = cv2.resize(n1.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_CUBIC) - 0.5
        n2 = C.fbm(wq, hq, wq / (90.0 * u), 3, seed=seed + 7)
        self.brush = cv2.resize(n2.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_CUBIC) - 0.5

    def rgba(self):
        a = self.acc[..., 3:4]
        rgb = self.acc[..., :3] / np.maximum(a, 1e-4)
        # bleed colour into the transparent margin (no dark fringes under bilinear sampling)
        solid = (a[..., 0] > 0.02).astype(np.float32)
        if solid.any():
            k = max(int(6 * self.u), 3)
            num = cv2.blur(rgb * solid[..., None], (2 * k + 1, 2 * k + 1))
            den = cv2.blur(solid, (2 * k + 1, 2 * k + 1))[..., None]
            fill = num / np.maximum(den, 1e-4)
            rgb = np.where(a > 0.02, rgb, fill)
        return dict(rgba=np.ascontiguousarray(np.concatenate([rgb, a], -1).astype(np.float32)),
                    ox=float(self.ox), oy=float(self.oy))


def smooth_normal(m, sigma, gain=2.2):
    """Normal of a heavily blurred silhouette, computed at reduced resolution and upsampled smoothly
    (no blocky steps in the terminator)."""
    h, w = m.shape
    q = max(1.0, sigma / 4.0)
    if q <= 1.5:
        return _grad_normal(cv2.GaussianBlur(m, (0, 0), sigma), gain * sigma)
    ws, hs = max(int(round(w / q)), 4), max(int(round(h / q)), 4)
    ms = cv2.resize(m, (ws, hs), interpolation=cv2.INTER_AREA)
    ms = cv2.GaussianBlur(ms, (0, 0), sigma / q)
    nx, ny = _grad_normal(ms, gain * sigma / q)
    return (cv2.resize(nx, (w, h), interpolation=cv2.INTER_CUBIC),
            cv2.resize(ny, (w, h), interpolation=cv2.INTER_CUBIC))


def _grad_normal(F, scale):
    gx = cv2.Sobel(F, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(F, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx, ny = -gx * scale, -gy * scale
    ln = np.sqrt(nx * nx + ny * ny)
    k = np.minimum(1.0, 1.0 / np.maximum(ln, 1e-6))
    return nx * k, ny * k


def paint_mass(P, m, x0, y0, base_y, hgt, sun, pal, st, group=None):
    """Paint one mass (anti-aliased silhouette m at plate px (x0, y0)) into plate P.
    base_y: plate-px row where the mass sinks into the fog; hgt: its visible height (px).
    sun: sun position in plate px; pal: row palette; st: style dict (see below).
    group: optional (Fbig, gx0, gy0) big-form field shared by several masses (towers)."""
    h, w = m.shape
    X0, Y0 = int(x0), int(y0)
    # clip to the plate
    cx0, cy0 = max(X0, 0), max(Y0, 0)
    cx1, cy1 = min(X0 + w, P.w), min(Y0 + h, P.h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    u = P.u
    ys, xs = np.mgrid[Y0:Y0 + h, X0:X0 + w].astype(np.float32)
    # ---- light direction (screen): up, and toward the sun's x
    Ld = np.array([st.get('lx', 0.0), -1.0], np.float32)
    Ld /= np.linalg.norm(Ld)
    # ---- big form normal (one coherent terminator) + mid normal (scallops of the big heads)
    if group is None:
        sb = max(st.get('form', 0.22) * hgt, 2.0)
        nbx, nby = smooth_normal(m, sb)
    else:
        nbx, nby = group
    sm = max(st.get('scallop_r', 0.06) * hgt, 1.2)
    nmx, nmy = smooth_normal(m, sm)
    hv = np.clip((base_y - ys) / max(hgt, 1.0), 0, 1.2)          # 0 at the fog line .. 1 at the top
    sl = (slice(cy0 - Y0, cy1 - Y0), slice(cx0 - X0, cx1 - X0))
    tear = np.zeros_like(m)
    brush = np.zeros_like(m)
    tear[sl] = P.tear[cy0:cy1, cx0:cx1]
    brush[sl] = P.brush[cy0:cy1, cx0:cx1]
    Lt = (st.get('wb', 0.55) * (nbx * Ld[0] + nby * Ld[1]) + st.get('wm', 0.22) * (nmx * Ld[0] + nmy * Ld[1])
          + st.get('wh', 0.5) * (hv - st.get('h0', 0.62)) + 0.1 * brush)
    # sun-axis proximity (plate x vs sun x): warmer, broader light near the axis
    prox = np.exp(-((xs - sun[0]) / (st.get('axis', 0.26) * st['W'])) ** 2).astype(np.float32)
    t1 = st.get('t1', 0.1) - st.get('lit_axis', 0.12) * prox
    e1 = st.get('e1', 0.035)
    # firm painted terminator: a hard threshold anti-aliased over ~1.5 px (+ a little softness far away)
    lit = K._blur((Lt > t1).astype(np.float32), max(0.9 * u, 0.6) + st.get('term_soft', 0.0) * hgt)
    # mid plane: a second, softer painted step below the terminator (its own value plane, not a smudge)
    mid = K._blur((Lt > t1 - st.get('mid_w', 0.12)).astype(np.float32), 1.5 * u + st.get('mid_soft', 0.012) * hgt)
    hot = _ss(t1 + 0.12, t1 + 0.42, Lt) * (0.35 + 0.65 * prox)
    p3 = prox[..., None]
    lit_c = _lerp(pal['lit_off'], pal['lit'], p3)
    mid_c = _lerp(pal['mid_off'], pal['mid'], p3)
    sh_c = _lerp(pal['shade_off'], pal['shade'], p3)
    if st.get('lit_grad', 0.0) > 0:
        # the lower part of each lobe's lit face turns rosy: the lobe in front then cuts a crisp,
        # light-on-mid overlapping edge across it (the painted 'stacked heads' read)
        lit_c = _lerp(lit_c, _lerp(lit_c, mid_c, 0.6), (st['lit_grad'] * _ss(0.7, 0.05, hv))[..., None])
    # shadow: deeper just under the terminator, reflected sky-light lower down
    refl = _ss(0.75, 0.1, hv)[..., None] * st.get('refl', 0.45)
    col = _lerp(sh_c, pal['refl'], refl)
    col = _lerp(col, mid_c, mid[..., None] * st.get('mid_amt', 0.85))
    col = _lerp(col, lit_c, lit[..., None])
    col = _lerp(col, pal['hi'], (hot * lit)[..., None] * st.get('hot', 0.8))
    # ---- torn, wind-combed base: colour melts into the valley haze, then alpha tears away
    fog_c = _lerp(pal['fog'], pal['fog_sun'], p3)
    fz = st.get('fog_z', 0.5)
    fy = (ys - (base_y - fz * hgt)) / max(fz * hgt, 1.0) + st.get('tear', 0.9) * tear * 2.0
    fogw = _ss(0.0, 1.0, fy) * st.get('fog_amt', 0.9)
    col = _lerp(col, fog_c, fogw[..., None])
    a = m if st.get('solid_base') else m * (1.0 - _ss(0.72, 1.25, fy))
    if st.get('haze', 0.0) > 0:
        col = _lerp(col, fog_c, st['haze'])
    # ---- rim: continuous, only where the silhouette faces the sun; width tapers to nothing away from it
    rim_k = st.get('rim', 0.0)
    if rim_k > 0:
        mb = (m > 0.5).astype(np.uint8)
        D = cv2.distanceTransform(np.pad(mb, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        se = max(st.get('rim_r', 0.03) * hgt, 2.5 * u)
        nex, ney = smooth_normal(m, se)
        Ls = np.array([st.get('rx', 0.0), -1.0], np.float32)
        Ls /= np.linalg.norm(Ls)
        face = _ss(st.get('face0', 0.25), 0.8, nex * Ls[0] + ney * Ls[1])
        rp = st.get('rim_px', 3.0) * u * (0.35 + 0.65 * prox) * face
        band = np.clip(rp + 0.5 - D, 0, 1) * (rp > 0.25) * m
        band *= (1.0 - _ss(0.1, 0.6, fy)) * rim_k
        rc = _lerp(RIM_SILVER, RIM_GOLD, (prox ** 0.7)[..., None])
        glow = np.exp(-D / max(4.0 * st.get('rim_px', 3.0) * u, 1.0)) * face * (0.2 + 0.5 * prox) * rim_k
        glow *= (1.0 - _ss(0.0, 0.5, fy)) * st.get('glow', 0.6)
        col = col + glow[..., None] * (rc - col) * 0.35
        col = _lerp(col, rc, band[..., None])
    a = np.clip(a, 0, 1)
    if rim_k > 0 and st.get('halo', 0.0) > 0:
        # soft glow spilling OUTSIDE the sun-facing silhouette (backlit crest glowing against what is behind)
        Do = cv2.distanceTransform(np.pad(1 - mb, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        ho = np.exp(-Do / max(3.0 * st.get('rim_px', 3.0) * u, 1.0)) * (1 - m) * face * (0.3 + 0.7 * prox)
        ho *= rim_k * st['halo'] * (1.0 - _ss(-0.2, 0.3, fy))
        col = np.where((m < 0.5)[..., None], rc, col)
        a = np.maximum(a, ho)
    acc = P.acc[cy0:cy1, cx0:cx1]
    a_ = a[sl][..., None]
    acc[..., :3] = col[sl] * a_ + acc[..., :3] * (1 - a_)
    acc[..., 3:4] = a_ + acc[..., 3:4] * (1 - a_)


def outer_rim(P, U, rx, rim_px, y_fade, se):
    """Thin bright lining along the plate's OUTER silhouette only (where the cloud meets open sky), and
    only where it faces the light (rx: horizontal lean of the light direction); fades out toward the base."""
    Ls = np.array([rx, -1.0], np.float32)
    Ls /= np.linalg.norm(Ls)
    nex, ney = smooth_normal(U, se)
    face = _ss(0.35, 0.85, nex * Ls[0] + ney * Ls[1])
    D = cv2.distanceTransform(np.pad((U > 0.5).astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    ys = np.arange(P.h, dtype=np.float32)[:, None]
    rp = rim_px * face * _ss(y_fade, y_fade - 0.25 * P.h, ys)
    band = np.clip(rp + 0.5 - D, 0, 1) * (rp > 0.25) * np.clip(U, 0, 1)
    glow = np.exp(-D / (4.0 * rim_px)) * face * 0.25 * _ss(y_fade, y_fade - 0.25 * P.h, ys)
    a = P.acc[..., 3]
    rgb = P.acc[..., :3] / np.maximum(a, 1e-4)[..., None]
    rc = np.array([1.55, 1.3, 0.92], np.float32)
    rgb = rgb + glow[..., None] * (rc - rgb) * 0.5
    rgb = rgb + (rc - rgb) * band[..., None]
    P.acc[..., :3] = rgb * a[..., None]


# ------------------------------------------------------------------------------------------ shapes
def heap_poly(cx, base_y, w, h, rng, flat=0.0):
    """Outline of a heaped mass: a dome envelope with a lumpy crown (2-4 big heads), flat base below."""
    n = 140
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    top = np.full(n, base_y, np.float32)
    k = rng.integers(2, 5)
    cs = np.sort(rng.uniform(-0.32, 0.32, k)) * w + cx
    for i, c in enumerate(cs):
        bw = w * rng.uniform(0.28, 0.55)
        bh = h * rng.uniform(0.6, 1.0) * (1.0 if i == int(np.argmin(np.abs(cs - cx))) else rng.uniform(0.6, 0.95))
        bh = min(bh, 0.8 * bw)
        d = np.clip(1 - ((xs - c) / (bw / 2)) ** 2, 0, 1)
        top = np.minimum(top, base_y - bh * d ** 0.8)
    env = np.clip(1 - ((xs - cx) / (w / 2)) ** 2, 0, 1) ** 0.6
    top = base_y - (base_y - top) * env
    if flat:
        top = np.maximum(top, base_y - h * (1 - flat))
    upper = np.stack([xs, top], 1)
    bowl = base_y + 0.4 * h * np.sqrt(np.clip(1 - ((xs - cx) / (w / 2)) ** 2, 0, 1))
    lower = np.stack([xs[::-1], bowl[::-1]], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


def mass_mask(poly, rng, size, detail=1.0, max_px=2400, down_cut=0.05):
    lv = ((0.07 * detail + 0.03, 0.16 * detail + 0.04, 0.85, 1.2, 2.4, 0.2, 0.5),
          (0.028, 0.06, 0.75, 1.1, 2.4), (0.012, 0.024, 0.45, 1.4, 3.2))
    if size < 40:
        lv = lv[:2]
    if size < 16:
        lv = lv[:1]
    return K.cauliflower(poly, rng, size, levels=lv, down_cut=down_cut, side_scale=0.55, ss=2 if size < 500 else 1,
                         concave=0.7, max_px=max_px)


# ------------------------------------------------------------------------------------------ camera / world
class Camera:
    """Pinhole camera over the deck. p = eased shot progress 0..1."""

    def __init__(self, W, H, hy, dolly=0.55, crane=0.12, truck=0.08, focal=1.0):
        self.W, self.H, self.HY, self.CX = W, H, hy, 0.5 * W
        self.F = focal * W
        self.dolly, self.crane, self.truck = dolly, crane, truck

    def state(self, p):
        return self.dolly * p, 1.0 + self.crane * p, self.truck * p       # zc, hc, camx

    def project(self, X, h, Z, p=1.0):
        zc, hc, cx = self.state(p)
        z = Z - zc
        return self.CX + self.F * (X - cx) / z, self.HY + self.F * (hc - h) / z, self.F / z

    def place(self, Z, p, pref=1.0):
        """(zoom, pivot, tx, ty) taking a plate painted with the camera at pref to the camera at p."""
        zc1, hc1, cx1 = self.state(pref)
        zc, hc, cx = self.state(p)
        z1, z = Z - zc1, Z - zc
        return z1 / z, (self.CX, self.HY), self.F * (cx1 - cx) / z, self.F * (hc - hc1) / z


ROWS_Z = (1.62, 2.05, 2.6, 3.25, 4.0, 4.9, 6.0, 7.3, 8.9, 10.8, 13.2, 16.0, 19.5, 24.0, 29.5, 36.0, 45.0, 56.0, 72.0)
Z_MIN, Z_MAX = ROWS_Z[0], ROWS_Z[-1]


def dist01(Z):
    return float(np.clip(math.log(Z / Z_MIN) / math.log(Z_MAX / Z_MIN), 0, 1))


def _plate_for(masses, u, seed, pad):
    xs0 = min(mm['x'] - mm['w'] * 0.62 for mm in masses) - pad
    xs1 = max(mm['x'] + mm['w'] * 0.62 for mm in masses) + pad
    ys0 = min(mm['base'] - mm['h'] * 1.25 for mm in masses) - pad
    ys1 = max(mm['base'] + mm['h'] * 0.6 for mm in masses) + pad
    return Plate(math.floor(xs0), math.floor(ys0), math.ceil(xs1 - xs0), math.ceil(ys1 - ys0), seed, u)


def ref_p(cam, Z):
    """Camera progress the plate of depth Z is painted for (near rows: mid-shot, so they are neither
    enormous plates nor blurry upsamples)."""
    return 1.0 if Z - cam.dolly > 2.5 else 0.4


def build_row(cam, Z, sun_ref, seed):
    """One row of heaped masses at depth Z, painted for the camera at ref_p. Returns plate dict (+ 'Z')."""
    W, H = cam.W, cam.H
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    a = dist01(Z)
    pal = row_palette(a)
    pr = ref_p(cam, Z)
    half = 0.5 * W * Z / cam.F                       # visible half-width (world) at the start of the shot
    sz = 1.0 if Z < 7 else (Z / 7.0) ** 0.55         # far rows: larger cells (they read as low bands)
    x = -half - 0.6 * sz - rng.uniform(0, 1.0) * sz
    xend = half + cam.truck + 0.8 * sz
    masses = []
    while x < xend:
        wW = sz * rng.uniform(0.55, 1.5)
        asp = rng.uniform(0.3, 0.5) * (1.0 - 0.35 * a)
        hW = wW * asp
        Zm = Z * (1 + rng.uniform(-0.07, 0.07))
        hb = rng.uniform(-0.08, 0.04) * sz
        if rng.random() > 0.16 or Z > 20:
            X = x + wW * 0.5
            sx, by, k = cam.project(X, hb, Zm, pr)
            masses.append(dict(x=sx, base=by, w=wW * k, h=hW * k, Z=Zm, seed=int(rng.integers(1 << 30))))
        x += wW * rng.uniform(0.42, 0.78)
    masses.sort(key=lambda mm: -mm['Z'])
    pad = 8 * u
    P = _plate_for(masses, u, seed + 3, pad)
    sun = (sun_ref[0] - P.ox, sun_ref[1] - P.oy)
    near = 1.0 - a
    for mm in masses:
        r2 = np.random.default_rng(mm['seed'])
        cx, by = mm['x'] - P.ox, mm['base'] - P.oy
        poly = heap_poly(cx, by, mm['w'], mm['h'], r2)
        m, x0, y0 = mass_mask(poly, r2, mm['h'], detail=1.0)
        lx = 0.7 * float(np.clip((sun[0] - cx) / (0.45 * W), -1, 1))
        prox_c = math.exp(-((cx - sun[0]) / (0.3 * W)) ** 2)
        st = dict(W=W, lx=lx, rx=1.3 * lx, form=0.2, scallop_r=0.07, wb=0.6, wm=0.25, wh=0.4,
                  h0=0.6 - 0.08 * prox_c + r2.uniform(-0.08, 0.08),
                  t1=0.2 + r2.uniform(-0.06, 0.06), lit_axis=0.12, e1=0.03 + 0.04 * a,
                  term_soft=0.004 + 0.02 * a, mid_w=0.1, refl=0.35, mid_amt=0.8,
                  fog_z=(0.42 + 0.1 * a) * r2.uniform(0.8, 1.25), tear=0.9 * near + 0.2, fog_amt=0.88,
                  rim=float(_ss(0.62, 0.3, a)), rim_px=1.4 + 2.8 * near ** 1.2, rim_r=0.035,
                  glow=0.7, hot=0.8, haze=0.3 * a ** 1.6, face0=0.3, halo=0.45)
        paint_mass(P, m, x0, y0, by, mm['h'], sun, pal, st)
    d = P.rgba()
    d['Z'] = Z
    d['pref'] = pr
    return d


def tower(cam, X, Z, height_w, sun_ref, seed, side=1.0, width=1.0, haze=0.0):
    """A towering cumulus rising out of the deck at (X, Z): stacked heaped masses painted back to front
    with ONE shared big-form normal (a single coherent terminator: warm-white sunward face, cool shadow
    mass on the far side, crisp only where a front mass overlaps), florets on the silhouette only and a
    thin bright rim only on the sunward (inner) edge. side=+1: sun to the right."""
    W = cam.W
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    pal = row_palette(dist01(Z) * 0.55)
    pal['lit'] = _c('#fff0d8')
    pal['lit_off'] = _c('#fbe3d0')
    pal['hi'] = np.array([1.0, 0.97, 0.9], np.float32)
    pal['mid'] = _c('#e4b8c8')
    pal['mid_off'] = _c('#d2b2d0')
    pal['shade'] = _c('#9c98d4')
    pal['shade_off'] = _c('#9496d0')
    pal['refl'] = _c('#b4b0e0')
    sx, by, k = cam.project(X, 0.0, Z)
    Hp = height_w * k
    # pieces (dx, dbase, w, h) as fractions of the tower height, painted top (farthest) first: rounded
    # heaps stacked up the tower; each lower heap overlaps the body above it -> crisp overlapping edges
    pieces = [(-0.04 * side, -0.68, 0.5, 0.32), (0.1 * side, -0.56, 0.52, 0.3), (-0.14 * side, -0.46, 0.56, 0.32),
              (0.04 * side, -0.34, 0.72, 0.34), (0.2 * side, -0.2, 0.6, 0.32), (-0.16 * side, -0.14, 0.66, 0.32),
              (0.06 * side, 0.0, 0.98, 0.32)]
    pad = 0.3 * Hp
    x0p, y0p = int(sx - 0.7 * Hp * width - pad), int(by - 1.1 * Hp - pad)
    P = Plate(x0p, y0p, int(1.4 * Hp * width + 2 * pad), int(1.3 * Hp + 2 * pad), seed + 5, u)
    cxp, byp = sx - P.ox, by - P.oy
    masks = []
    for (dx, db, fw, fh) in pieces:
        cx = cxp + dx * Hp * width + rng.uniform(-0.02, 0.02) * Hp
        b = byp + db * Hp
        w_, h_ = fw * Hp * width * rng.uniform(0.92, 1.08), fh * Hp * rng.uniform(0.9, 1.1)
        poly = heap_poly(cx, b, w_, h_, rng)
        m, mx0, my0 = mass_mask(poly, rng, h_, detail=1.2, down_cut=0.45)
        masks.append((m, mx0, my0, b, h_))
    U = np.zeros((P.h, P.w), np.float32)
    for (m, mx0, my0, b, h_) in masks:
        hh, ww = m.shape
        a0, b0 = max(my0, 0), max(mx0, 0)
        a1, b1 = min(my0 + hh, P.h), min(mx0 + ww, P.w)
        U[a0:a1, b0:b1] = np.maximum(U[a0:a1, b0:b1], m[a0 - my0:a1 - my0, b0 - mx0:b1 - mx0])
    sb = 0.16 * Hp * width
    gnx, gny = smooth_normal(U, sb, 2.4)
    sun = (sun_ref[0] - P.ox, sun_ref[1] - P.oy)
    for i, (m, mx0, my0, b, h_) in enumerate(masks):
        hh, ww = m.shape
        gx = np.zeros_like(m)
        gy = np.zeros_like(m)
        a0, b0 = max(my0, 0), max(mx0, 0)
        a1, b1 = min(my0 + hh, P.h), min(mx0 + ww, P.w)
        gx[a0 - my0:a1 - my0, b0 - mx0:b1 - mx0] = gnx[a0:a1, b0:b1]
        gy[a0 - my0:a1 - my0, b0 - mx0:b1 - mx0] = gny[a0:a1, b0:b1]
        last = i == len(masks) - 1
        st = dict(W=W, lx=1.4 * side, rx=2.0 * side, wb=0.8, wm=0.12, wh=0.06, lit_grad=0.55, mid_w=0.14, h0=0.5, t1=0.0, lit_axis=0.0,
                  e1=0.025, scallop_r=0.06, refl=0.5, mid_amt=0.7, fog_z=0.45 if last else 0.3, tear=0.6,
                  fog_amt=0.8 if last else 0.0, solid_base=not last,
                  rim=0.0, rim_px=2.2, rim_r=0.03, glow=0.4, hot=0.7, haze=haze, face0=0.45, axis=10.0)
        paint_mass(P, m, mx0, my0, b, h_, sun, pal, st, group=(gx, gy))
    outer_rim(P, U, 1.7 * side, 2.4 * u, byp - 0.1 * Hp, 0.03 * Hp)
    d = P.rgba()
    d['Z'] = Z
    return d


def fg_wisp(cam, X, h, Z, w_w, h_w, seed):
    """A torn, semi-transparent cloud wisp close to the camera, lit along its upper edge."""
    W = cam.W
    u = W / 1920.0
    sx, sy, k = cam.project(X, h, Z, 0.0)
    wp, hp = w_w * k, h_w * k
    P = Plate(int(sx - 0.7 * wp), int(sy - 1.2 * hp), int(1.4 * wp), int(2.2 * hp), seed, u)
    ys, xs = np.mgrid[0:P.h, 0:P.w].astype(np.float32)
    cx, cy = sx - P.ox, sy - P.oy
    r = np.sqrt(((xs - cx) / (0.5 * wp)) ** 2 + ((ys - cy) / (0.5 * hp)) ** 2)
    n = C.fbm(max(P.w // 4, 8), max(P.h // 4, 8), 6.0, 5, seed=seed)
    n = cv2.resize(n.astype(np.float32), (P.w, P.h), interpolation=cv2.INTER_CUBIC)
    s2 = C.fbm(max(P.w // 4, 8), max(P.h // 12, 4), 14.0, 4, seed=seed + 1)
    s2 = cv2.resize(s2.astype(np.float32), (P.w, P.h), interpolation=cv2.INTER_CUBIC)
    dens = np.clip((1.0 - r) * 1.4 + (n - 0.5) * 1.2 + (s2 - 0.5) * 0.8, 0, 1)
    dens = K._blur(dens, 3.0 * u) ** 1.3
    top = np.clip((cy - ys) / (0.5 * hp) + 0.3, 0, 1)
    pal = row_palette(0.0)
    col = np.broadcast_to(_lerp(pal['fog'], pal['refl'], 0.4), dens.shape + (3,))
    col = _lerp(col, _c('#fbe4d2'), (top ** 1.2)[..., None])
    P.acc[..., :3] = col * dens[..., None] * 0.8
    P.acc[..., 3] = dens * 0.8
    d = P.rgba()
    d['Z'] = Z
    d['pref'] = 0.0
    return d
