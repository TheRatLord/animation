"""Hero cumulonimbus + far bank for s03_city_dusk (round 7), painted with the shared lib.clouds2 module.

Round-7 art direction (fixing round 6's 'flat vector cut-out' read):
  * 4 big painted value masses (hot pink-gold lit face toward the afterglow at centre-left, magenta mid,
    cool violet shadow, a deeper indigo core low down) + 2 small forward knots. Interiors stay large and
    smooth; the few crisp overlapping-mass edges come from the masses themselves (no inner lobes, no
    skylight steps, no shadow-side wrap -> no stacked scallops inside the violet);
  * fine, irregular cauliflower breakup on the SILHOUETTE only (5 floret levels, low clump);
  * lit face pushed to a hot pink-gold that goes near-white on the sun-facing crowns; a firm lit-on-lit
    step instead of a soft seam between lit and magenta;
  * a crisp 2-4 px hot-gold lining ONLY on the sun-facing (left / lower-left) silhouette, no glow halo;
    the shadow side (right) keeps at most a dim cool sky-bounce edge;
  * a torn, hazy base: the lowest ~20 % is eaten along wind-torn streaks and fades into the sky colour
    behind the skyline (no solid violet block behind the towers).
"""
import numpy as np
import cv2

from lib import core as C, clouds2 as K2

PAL = dict(hi=(1.0, 0.9, 0.8), lit=(1.0, 0.77, 0.63), lit_lo=(0.99, 0.64, 0.54), mid='#e07a8a',
           shade='#9c76b4', deep='#6e58a6', refl='#b096d2', bounce='#e0707e', rim=(1.65, 0.92, 0.26),
           haze='#b08cc8', edge_dark=0.0)
PAL_FAR = dict(hi=(1.0, 0.8, 0.72), lit='#f4a0a6', lit_lo='#e08aa8', mid='#c47aac', shade='#8a78c0',
               deep='#6c5eac', refl='#9486c8', bounce='#b478ac', rim=(1.4, 0.8, 0.72), haze='#a090d0',
               edge_dark=0.0)
# back shoulder behind the hero (the 'back lavender cumulus'): cooler, hazier, but a pink lit top
PAL_BACK = dict(hi=(1.0, 0.8, 0.8), lit='#f3a4b4', lit_lo='#d890b8', mid='#b884bc', shade='#8c80c6',
                deep='#6e66b4', refl='#9a90d0', bounce='#b884b8', rim=(1.5, 0.78, 0.8), haze='#a898d4',
                edge_dark=0.0)

# silhouette cauliflower: a few big heads, then CLUSTERS of florets separated by smooth stretches
# (high clump), no uniform fine notching
LEVELS = ((0.16, 0.34, 0.95, 1.3, 2.4, 0.05, 0.42), (0.06, 0.13, 0.85, 0.9, 2.2, 0.15, 0.5),
          (0.026, 0.05, 0.8, 0.9, 2.0, 0.25, 0.55), (0.011, 0.022, 0.55, 1.0, 2.4, 0.2, 0.5))
CLUMP = 0.75


def _mask(poly, rng, size, sun, levels=LEVELS, clump=CLUMP, ss=4):
    """Cauliflower mask grown at 4x supersampling (anti-aliased, no pixel-stepped notches)."""
    poly = np.asarray(poly, np.float32)
    sp = np.asarray(sun, np.float32) - poly.mean(0)
    sp = sp / (float(np.hypot(*sp)) + 1e-6)
    return K2.cauliflower(poly, rng, size, levels, down_cut=0.3, side_scale=0.6, sun_dir=sp, sun_bias=0.35,
                          clump=clump, concave=0.75, ss=ss)


def _tower(P, X, Y, H, seed, sun_key):
    rng = np.random.default_rng(seed)
    cx, base, Hc = X(0.84), Y(0.575), 0.48 * H
    g = (cx - 0.02 * Hc, base - 0.5 * Hc, 0.42 * Hc, 0.56 * Hc)
    # per-mass domes dominate (form low): every lobe is lit on its top-left and turns into shadow under
    # itself -> plane changes follow the lobes, no band hugging the cloud-wide silhouette
    common = dict(group=g, form=0.22, mass_r=0.42, smooth=0.07, sun_z=0.3, wrap=0.0, term=0.28, term_w=0.035,
                  rim_interior=0.0, term_noise=0.12, inner=0, sky=0.12, halo=0.0, rim=0.0, hot=0.9,
                  lit_step=0.35, scallop=0.7, firm=0.7, soften=0.45, rim_shadow=0.0, shade_top=0.05,
                  valley_dark=0.12, refl=0.4, bounce=0.35, base_dark=0.45, split=0.28)
    masses = [
        # B  back shoulder on the right (behind the hero; hazier, pink lit top, violet base)
        ('B', dict(cx=cx + 0.36 * Hc, base_y=base - 0.12 * Hc, width=0.44 * Hc, height=0.44 * Hc, power=2.2,
                   lump=0.14, lean=0.05, base_round=0.3), dict(size=0.3 * Hc, pal=PAL_BACK, haze=0.12,
                                                               split=0.22, sun_z=0.45)),
        # A  back crown: the tallest head, leaning left
        ('A', dict(cx=cx + 0.05 * Hc, base_y=base - 0.5 * Hc, width=0.38 * Hc, height=0.5 * Hc, power=2.3,
                   lump=0.13, lean=-0.07, base_round=0.3), dict(size=0.32 * Hc)),
        # D  main body, lower right, wide
        ('D', dict(cx=cx + 0.16 * Hc, base_y=base + 0.02 * Hc, width=0.5 * Hc, height=0.5 * Hc, power=2.4,
                   lump=0.12, skew=0.1, base_round=0.15), dict(size=0.3 * Hc, cast=0.3, merge=False, lit_bias=0.14)),
        # C  sunward lower-left lobe overlapping the crown's base (crisp lit-over-shadow edge)
        ('C', dict(cx=cx - 0.2 * Hc, base_y=base - 0.2 * Hc, width=0.36 * Hc, height=0.4 * Hc, power=2.3,
                   lump=0.12, lean=-0.05, base_round=0.3), dict(size=0.28 * Hc, cast=0.45)),
        # small forward knots low on each flank
        ('F', dict(cx=cx - 0.36 * Hc, base_y=base + 0.03 * Hc, width=0.3 * Hc, height=0.17 * Hc, power=2.4,
                   lump=0.1, base_round=0.12), dict(size=0.15 * Hc, haze=0.05)),
        ('G', dict(cx=cx + 0.44 * Hc, base_y=base + 0.04 * Hc, width=0.3 * Hc, height=0.17 * Hc, power=2.3,
                   lump=0.12, base_round=0.12), dict(size=0.15 * Hc, haze=0.08)),
    ]
    for name, ek, sk in masses:
        poly = K2.envelope(rng=rng, **ek)
        kw = dict(common)
        kw.update(sk)
        mrng = np.random.default_rng(int(rng.integers(1 << 30)))
        kw['mask'] = _mask(poly, mrng, kw['size'], sun_key)
        P.shape(None, rng=rng, **kw)
    return cx, base, Hc


def lining(P, sun, strength=1.0, rim_px=2.8, pal=None):
    """Gold lining on the outer silhouette facing the (low, left) afterglow: thick on convex sun-facing
    bulges, thinning to nothing in the concave notches between them; anti-aliased, no halo."""
    pal = K2.palette(PAL if pal is None else pal)
    u = P.u
    A = np.ascontiguousarray(np.clip(P.prem[..., 3], 0, 1))
    hh, ww = A.shape
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl

    def toward(img, r):
        return cv2.remap(img, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    # convexity at floret scale: the blurred mask is < 0.5 on bulges, > 0.5 in notches
    cv_ = K2._blur(A, 7.0 * u)
    convex = K2._ss(0.55, 0.42, cv_)
    # measured on the contour ring, then carried a few px inward (the rim's width is set at the edge)
    ring = np.clip(A - cv2.erode(A, np.ones((3, 3), np.uint8)), 0, 1)
    sg = 2.5 * rim_px * u
    convex = K2._blur(convex * ring, sg) / np.maximum(K2._blur(ring, sg), 1e-4)
    convex = np.clip(convex, 0, 1) * (K2._blur(ring, sg) > 1e-3)
    r_thin, r_thick = 0.45 * rim_px * u, 1.6 * rim_px * u
    rr = r_thin + (r_thick - r_thin) * convex
    core = A * (1 - cv2.remap(A, xs + ux * rr, ys + uy * rr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0))
    soft = A * (1 - K2._blur(toward(A, 2.6 * rim_px * u), rim_px * u))
    rgb = P.prem[..., :3] / np.maximum(A, 1e-5)[..., None]
    litn = K2._ss(0.42, 0.72, rgb.max(-1))     # round 14: the dusk rework darkened the lit side
    k = np.clip(core * 1.7 + 0.3 * soft * convex, 0, 1) * (0.1 + 0.9 * litn) * K2._ss(0.0, 0.6, convex)
    k = np.clip(k * strength, 0, 1)
    rim = np.asarray(pal['rim'], np.float32)
    # a warmer peach band just inside the lining (lined edges read as light against the cream face)
    band = np.clip(A * (1 - K2._blur(toward(A, 3.2 * rim_px * u), 1.2 * rim_px * u)) - k, 0, 1)
    band = band * litn * (0.4 + 0.6 * convex) * 0.75 * strength
    P.prem[..., :3] = P.prem[..., :3] * (1 - band[..., None]) + pal['lit_lo'] * (A * band)[..., None]
    P.prem[..., :3] = P.prem[..., :3] * (1 - k[..., None]) + rim * (A * k)[..., None]


def lobe_shade(tower, seed, u, light=(-0.62, -0.58, 0.52), k=0.55):
    """Round 13: lobe-on-lobe internal modelling (wwy_01). Packs the cloud with cauliflower lobes at three
    scales (big heads, mid lobes, small florets; smaller ones sit in front), z-buffers them as domes and lights
    each dome from the up-left key: every lobe gets its own warm-white lit cap and a soft blue-violet underside,
    and where a lobe overlaps the one behind it the edge reads (lit lobe over the shade of the next). Painted,
    not rendered: a firm but soft terminator and flat caps, applied over the existing value masses (the big
    violet shadow side stays in shadow; there lobes only get a faint reflected-light cap)."""
    A = tower[..., 3]
    hh, ww = A.shape
    rng = np.random.default_rng(seed + 991)
    m8 = (A > 0.5).astype(np.uint8)
    if m8.sum() < 100:
        return tower
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    # a pillow over the whole silhouette (every cauliflower bump of the outline becomes a lobe that turns into
    # its own shadow) + domes packed inside at two scales (big heads, mid lobes; the mid ones sit in front)
    Hf = np.sqrt(dist * (60.0 * u)).astype(np.float32)
    ys_i, xs_i = np.nonzero(m8)
    for (rmin, rmax, n, zoff) in ((40 * u, 100 * u, 45, 0.0), (16 * u, 36 * u, 90, 18 * u)):
        pick = rng.integers(0, len(xs_i), n * 4)
        cnt = 0
        for j in pick:
            if cnt >= n:
                break
            cx, cy = float(xs_i[j]), float(ys_i[j])
            dmax = dist[int(cy), int(cx)]
            r = float(rng.uniform(rmin, rmax))
            if dmax < 0.3 * r or dmax > 2.5 * r:
                continue
            cnt += 1
            x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, ww))
            y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, hh))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = (xx - cx) / r, (yy - cy) / r
            dy2 = np.where(dy > 0, dy * 1.3, dy)
            d2 = dx * dx + dy2 * dy2
            h = np.sqrt(np.clip(1.0 - d2, 0, 1)) * r * 0.8 + Hf[int(cy), int(cx)] * 0.9 + zoff
            h = np.where(d2 < 1.0, h, -1e9)
            np.maximum(Hf[y0:y1, x0:x1], h, out=Hf[y0:y1, x0:x1])
    Hs = cv2.GaussianBlur(Hf, (0, 0), 2.5 * u)
    gx = cv2.Sobel(Hs, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Hs, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx, ny, nz = -gx, -gy, np.ones_like(gx) * 0.9
    nl = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    covered = m8 > 0
    L = np.array(light, np.float32)
    L = L / np.linalg.norm(L)
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    # soft internal gradient + a firm (not hard) painted terminator
    s = C.smoothstep(-0.1, 0.7, ndl) * 0.6 + C.smoothstep(0.2, 0.42, ndl) * 0.4
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), 1.5 * u)
    rgb = tower[..., :3]
    lum = rgb.max(-1)
    litm = C.smoothstep(0.62, 0.9, cv2.GaussianBlur(lum, (0, 0), 6 * u))
    lit_c = np.array([1.02, 0.93, 0.82], np.float32)
    mid_c = np.array([0.96, 0.66, 0.66], np.float32)
    shd_c = np.array([0.6, 0.55, 0.8], np.float32)
    tgt = np.where((s < 0.5)[..., None], shd_c + (mid_c - shd_c) * (s / 0.5)[..., None],
                   mid_c + (lit_c - mid_c) * ((s - 0.5) / 0.5)[..., None])
    # keep the painted hue of the region (lit mass: warm cream / peach), lobes model its value
    tgt_lit = tgt * 0.7 + rgb * 0.3 * (0.75 + 0.5 * s[..., None])
    tgt_shd = rgb * (0.86 + 0.3 * s[..., None]) + np.array([0.04, 0.02, 0.06], np.float32) * s[..., None]
    new = tgt_lit * litm[..., None] + tgt_shd * (1 - litm[..., None])
    # protect the painted silhouette lining (outer few px) and the torn base
    inner = cv2.GaussianBlur(cv2.erode(m8, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 2.0 * u)
    kk = (k * inner * covered)[..., None]
    tower[..., :3] = rgb * (1 - kk) + new * kk
    return tower


def dusk_rework(tower, X, Y, H, u, seed, key=(-0.6, -0.8)):
    """Round 14 (reviewer: 'clay / plasticky, evenly sized bubbles'): push the painted cloud toward yn_02 /
    wwy_01 at dusk.
      * clusters of SMALL cauliflower florets along the lit up-left silhouette (varied sizes, bunched, with
        smooth stretches between clusters): each floret is a lit cap with a tucked violet underside;
      * value: the body is less cream (warm peach-rose lit side, deeper blue-violet shade, a darker cool core);
      * the lower third flattens and darkens into one cool violet mass (flat-ish base, few internal forms).
    Straight-alpha RGBA in, out."""
    rng = np.random.default_rng(seed + 1414)
    A = np.ascontiguousarray(tower[..., 3])
    rgb = tower[..., :3].copy()
    hh, ww = A.shape
    m8 = (A > 0.5).astype(np.uint8)
    if m8.sum() < 100:
        return tower
    # ---- value / hue: warm lit, cool deep shade
    lum = rgb.max(-1)
    lit = C.smoothstep(0.6, 0.92, cv2.GaussianBlur(lum, (0, 0), 3 * u))[..., None]
    shade_t = np.array([0.46, 0.4, 0.7], np.float32)
    rgb = rgb * (0.78 + 0.14 * lit) + (1 - lit) * (shade_t * 0.9 - rgb * 0.25) * 0.55
    rgb = rgb + lit * (np.array([0.02, -0.03, -0.06], np.float32))
    # ---- lower part: one flat, darker cool violet mass
    ys = np.arange(hh, dtype=np.float32)[:, None, None]
    bm = C.smoothstep(Y(0.3), Y(0.46), ys)
    base_c = np.array([0.4, 0.33, 0.6], np.float32)
    lum2 = rgb.mean(-1, keepdims=True)
    flat = base_c * (0.75 + 0.5 * lum2)
    rgb = rgb * (1 - 0.72 * bm) + flat * 0.72 * bm
    # ---- florets on the lit silhouette
    Ab = cv2.GaussianBlur(A, (0, 0), 3.0 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kx, ky = key
    add_a = np.zeros_like(A)
    add_c = np.zeros_like(rgb)
    ss = 3
    big_a = np.zeros((hh * 1, ww * 1), np.float32)
    for cnt in cnts:
        pts = cnt[:, 0, :]
        n = len(pts)
        if n < 60:
            continue
        # outward normals, facing the key light?
        g = np.stack([gx[pts[:, 1], pts[:, 0]], gy[pts[:, 1], pts[:, 0]]], 1)
        gl = np.linalg.norm(g, axis=1) + 1e-6
        nx, ny = -g[:, 0] / gl, -g[:, 1] / gl
        face = nx * kx + ny * ky
        ok = np.nonzero((face > 0.2) & (pts[:, 1] < Y(0.44)))[0]
        if len(ok) == 0:
            continue
        ncl = int(len(ok) / (26 * u)) + 1
        centres = rng.choice(ok, size=min(ncl, len(ok)), replace=False)
        for c0 in centres:
            if rng.random() < 0.35:
                continue            # smooth stretches between the clusters
            nf = int(rng.integers(3, 8))
            step = 0.0
            for j in range(nf):
                r = float(rng.uniform(3.0, 9.0) * u * (1.0 - 0.45 * j / nf))
                step += r * rng.uniform(1.0, 1.6)
                idx = int(c0 + (step if j % 2 == 0 else -step) / 1.0) % n
                px, py = float(pts[idx, 0]), float(pts[idx, 1])
                fx, fy = nx[idx], ny[idx]
                cx, cy = px - fx * r * 1.05, py - fy * r * 1.05     # inside the edge: modelling only
                x0, x1 = int(max(cx - r - 3, 0)), int(min(cx + r + 4, ww))
                y0, y1 = int(max(cy - r - 3, 0)), int(min(cy + r + 4, hh))
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                dx, dy = (xx - cx) / r, (yy - cy) / r
                d = np.sqrt(dx * dx + dy * dy)
                # the tuck: a thin crescent of cool shade under / behind each floret (the lobe edge reads
                # against the lobe behind it); a faint warm cap toward the key
                sh = np.clip(dx * kx + dy * ky, -1, 1)
                ring = np.clip((1.0 - d) * r + 0.5, 0, 1) * np.clip((d - 0.72) * r * 0.8, 0, 1)
                tuck = ring * C.smoothstep(0.0, -0.5, sh)
                cap = np.clip((1.0 - d) * r + 0.5, 0, 1) * C.smoothstep(0.1, 0.7, sh) * 0.5
                sub_a = add_a[y0:y1, x0:x1]
                sub_c = add_c[y0:y1, x0:x1]
                np.maximum(sub_a, tuck, out=sub_a)
                np.maximum(sub_c[..., 0], cap, out=sub_c[..., 0])
    inside = C.smoothstep(0.5, 0.9, A)[..., None]
    add_a[:] = 0.0          # round 14b: the floret tucks read as bubble rings; value / base rework only
    add_c[:] = 0.0
    tk = (add_a[..., None] * inside * 0.55 * (1 - bm))
    rgb = rgb * (1 - tk) + np.array([0.56, 0.44, 0.7], np.float32) * tk
    cp = add_c[..., :1] * inside * (1 - bm) * (1 - add_a[..., None])
    rgb = rgb + cp * np.array([0.12, 0.07, 0.03], np.float32)
    a_new = A
    tower[..., :3] = np.clip(rgb, 0, None)
    tower[..., 3] = a_new
    return tower


def sun_rim(tower, sun, u, width=2.4, col=(1.75, 0.66, 0.46)):
    """Round 14: crisp hot orange-pink rim on the silhouette edges that face the real (low, left) sun
    (yn_02 / wwy_01): 1-3 px core measured by shifting the mask toward the sun, strongest on the edges that
    look straight at it, plus a softer peach-rose band a few px inside."""
    A = np.ascontiguousarray(tower[..., 3])
    hh, ww = A.shape
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl
    sh = lambda r: cv2.remap(A, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    core = np.clip(A - sh(width * u), 0, 1)
    band = np.clip(A - cv2.GaussianBlur(sh(6.0 * u), (0, 0), 2.0 * u), 0, 1) * (1 - core)
    # lower on the cloud = closer to the horizon glow = hotter
    hot = C.smoothstep(0.05 * hh, 0.5 * hh, ys)
    k = np.clip(core * 1.3, 0, 1) * (0.55 + 0.45 * hot)
    kb = band * 0.45 * (0.5 + 0.5 * hot)
    rgb = tower[..., :3]
    rgb[:] = rgb * (1 - kb[..., None]) + np.array([1.2, 0.66, 0.6], np.float32) * kb[..., None]
    rgb[:] = rgb * (1 - k[..., None]) + np.asarray(col, np.float32) * k[..., None]
    return tower


def _torn_base(tower, X, Y, H, sky, seed):
    """Tear the lowest part of the tower along wind-streaks and dissolve it into the sky colour."""
    ph, pw = tower.shape[:2]
    ys = np.arange(ph, dtype=np.float32)[:, None]
    rng_n = K2._noise(pw, ph, 18.0, seed + 31, 4, stretch=6.0)
    rng_f = K2._noise(pw, ph, 60.0, seed + 37, 2, stretch=8.0)
    prof = K2._noise(pw, 8, 9.0, seed + 41, 3)[4][None, :]
    # the cut line wanders along x (round 14: flatter base, as a dusk cumulonimbus sits on a level floor)
    y_cut = Y(0.475) + (prof - 0.5) * 0.035 * H
    d = (ys - y_cut) / (0.07 * H)
    n = 0.65 * rng_n + 0.35 * rng_f
    keep = C.smoothstep(0.55, -0.35, d + 0.9 * (n - 0.5) * 2.0 * C.smoothstep(-1.2, 0.4, d))
    keep = np.clip(keep, 0, 1).astype(np.float32)
    a = tower[..., 3] * keep
    # fading part takes the sky's colour (aerial haze at the horizon, the sky shows through)
    fz = C.smoothstep(Y(0.36), Y(0.56), ys)[..., None] * 0.6
    if sky is not None:
        hz = cv2.GaussianBlur(sky, (0, 0), 0.01 * H)[..., :3] * 0.8 + np.array([0.55, 0.42, 0.78], np.float32) * 0.2
    else:
        hz = np.array([0.62, 0.46, 0.8], np.float32)
    tower[..., :3] = tower[..., :3] * (1 - fz) + hz * fz
    tower[..., 3] = a
    return tower


def paint(pw, ph, W, H, ox, sun, seed=5, sky=None, key=(-0.56, -0.83), rim_y=0.33):
    """Straight-alpha RGBA plates (ph, pw, 4): (tower, bank). Frame fractions offset by ox px."""
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    u = W / 1920.0
    # painted key: high up-left (tops lit, shadow tucked under every lobe); the lining looks at the real
    # afterglow low on the left
    sun_l = (X(0.84) + key[0] * 3 * H, Y(0.3) + key[1] * 3 * H)
    P = K2.Painter(pw, ph, sun_l, PAL, u, seed=seed + 1, sun_z=0.42)
    cx, base, Hc = _tower(P, X, Y, H, seed, sun_l)
    top = base - Hc
    tower = P.rgba()
    tower = lobe_shade(tower, seed, u)
    tower = dusk_rework(tower, X, Y, H, u, seed)
    # the lining goes on after the lobe modelling (round 13: stronger, pinker-orange on the sun-facing lobes)
    a_ = tower[..., 3:4]
    P.prem = np.concatenate([tower[..., :3] * a_, a_], -1).astype(np.float32)
    lining(P, (float(sun[0]), Y(rim_y)), 1.7, rim_px=2.6, pal=dict(PAL, rim=(1.95, 0.7, 0.45)))
    tower = P.rgba()
    tower = sun_rim(tower, (float(sun[0]), float(sun[1])), u)
    tower = _torn_base(tower, X, Y, H, sky, seed)
    # ------------------------------------------------------------------ low far bank behind the skyline
    Pb = K2.Painter(pw, ph, (float(sun[0]), float(sun[1])), PAL_FAR, u, seed=seed + 5, sun_z=0.05)
    for i, (fx, fy, cw, ch) in enumerate(((0.64, 0.585, 0.16, 0.07), (0.98, 0.59, 0.3, 0.14),
                                         (1.1, 0.59, 0.18, 0.09), (0.76, 0.59, 0.14, 0.06))):
        K2.cumulus(Pb, X(fx), Y(fy), cw * W, ch * H, seed=seed * 10 + i, pal=PAL_FAR, n=3, haze=0.3)
    Pb.lining(0.8, rim_px=2.2, halo=0.0)
    Pb.haze_band(Y(0.47), Y(0.6), np.array([0.62, 0.5, 0.84], np.float32), 0.55)
    bank = Pb.rgba()
    return tower.astype(np.float32), bank.astype(np.float32)


def bloom_src(img, cloud_a, k=0.6):
    """Bloom SOURCE with the painted cloud interior held down (the lit face must not bloom into an
    airbrushed glow outline); the outer ~5 px (the HDR gold lining) keeps its full value."""
    a = np.ascontiguousarray(cloud_a, dtype=np.float32)
    a = cv2.GaussianBlur(cv2.erode(a, np.ones((11, 11), np.uint8)), (0, 0), 2.0)[..., None]
    return img * (1 - k * np.clip(a, 0, 1))
