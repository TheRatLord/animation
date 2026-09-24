"""s01 helper: the hero cumulonimbus, painted with lib/clouds2 (Painter, mass by mass).

Summer afternoon, the sun high up-LEFT, sitting just behind the anvil's blunt upper-left crest.
One continuous rising column (no stacked tiers): masses overlap vertically with staggered heights and
offsets, the column leans and billows out to the RIGHT (shadow side) in 2-3 big lobes, and the anvil
grows out of the crown (no separate cap, no waist) and streams downwind to the right into fibres.

Light: key from the upper left, a little in front. Warm cream / near-white on the sun-facing upper-left
of every mass and the whole left flank; cool lavender only in narrow wedges under overhangs on the
right / lower side; crisp lit-on-lit overlap edges. The silver/gold lining is direction-aware (only on
the sun-facing top-left edges + the backlit crest near the sun); shadowed right / lower edges meet the
sky softly. The lower third loses contrast and saturation and tears into wisps that melt into a pale
cyan horizon haze (aerial perspective)."""
import numpy as np
import cv2

from lib import clouds2 as K

PAL = dict(hi=(1.0, 0.99, 0.965), lit='#fbf6ee', lit_lo='#dedbe6', mid='#aeb0d0', shade='#7482b8',
           deep='#5f6da8', refl='#8e9dd0', bounce='#bdbad2', rim=(1.7, 1.55, 1.2), haze='#cfe6f7',
           edge_dark=0.0)
PAL_BASE = dict(PAL, lit='#f1eeec', lit_lo='#d6d9e8', hi=(0.98, 0.975, 0.965), shade='#8a97c8',
                deep='#7b89c0', refl='#a3b2de', bounce='#c8c8da')
PAL_ANVIL = dict(PAL, lit='#fbf6ee', lit_lo='#e4e3ee', mid='#c0c2dc', shade='#8a98cc', deep='#7886c2',
                 refl='#a6b4e2')
LEV_BIG = ((0.14, 0.4, 0.95, 1.2, 2.4, 0.12, 0.45), (0.05, 0.13, 0.85, 1.0, 2.2, 0.3, 0.6),
           (0.02, 0.045, 0.6, 1.2, 3.0), (0.012, 0.02, 0.2, 1.6, 4.0))


def sun_rim(rgba, sun, key_dir, W, top_y, base_y, strength=1.0):
    """Direction-aware lining: a crisp 2-4 px gold/white line ONLY where the outer silhouette faces the
    key (upper-left edges of the upper tower) and round the backlit crest near the sun; shadowed right
    and lower edges get nothing (they meet the sky softly)."""
    u = W / 1920.0
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    h, w = A.shape
    Ab = cv2.GaussianBlur(A, (0, 0), 11 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    Hc = base_y - top_y
    d = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    near = np.exp(-(d / 0.2) ** 2)
    kd = np.asarray(key_dir, np.float32)
    kd = kd / float(np.hypot(*kd))
    face = K._ss(-0.05, 0.6, nx * kd[0] + ny * kd[1])
    vz = 1 - K._ss(top_y + 0.2 * Hc, top_y + 0.47 * Hc, ys)
    k_face = face * vz * (0.5 + 0.5 * np.exp(-d / 0.45))
    k = np.clip(np.maximum(k_face, near * np.clip(0.2 + 0.8 * (-ny), 0, 1)), 0, 1)
    dI = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    core = K._ss(0.0, 0.8 * u, dI) * K._ss(3.6 * u, 2.2 * u, dI)
    wide = cv2.GaussianBlur(K._ss(8 * u, 2 * u, dI) * (A > 0.5), (0, 0), 1.5 * u)
    ring = np.maximum(core, wide * (0.25 + 0.45 * near))
    kk = np.clip(ring * k * strength, 0, 1)
    white = np.array([1.6, 1.46, 0.95], np.float32)
    gold = np.array([1.9, 1.4, 0.7], np.float32)
    gw = np.clip(0.3 + near * 1.3, 0, 1)[..., None]
    col = white * (1 - gw) + gold * gw
    rgb = rgba[..., :3] * (1 - kk[..., None]) + col * kk[..., None]
    hk = cv2.GaussianBlur(kk, (0, 0), 2.5 * u) * 0.7 + cv2.GaussianBlur(kk, (0, 0), 9 * u) * 0.45
    hk = np.clip(hk * (1 - A) * (0.35 + 0.9 * near), 0, 1)
    a = np.clip(A + hk * 0.4, 0, 1)
    rgb = (rgb * A[..., None] + col * hk[..., None] * 0.5) / np.maximum(a, 1e-4)[..., None]
    rgb = np.where(a[..., None] > 1e-4, rgb, rgba[..., :3])
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32)


def repaint(rgba, sun, key_dir, W, top_y, base_y):
    """Art-direction pass on the painted tower (values only, silhouette untouched):
    * one-sided edges: shadow boundaries whose LIT side faces the sun are the terminator of the same
      mass -> softened into a short painted gradient; boundaries whose lit side faces away (a lit lobe
      in front cutting across a shadow plane) stay crisp. So each shadow mass has a crisp edge on one
      side only and reads as a mass, not a punched hole;
    * brilliant warm-white lit faces (warmer / hotter toward the sun, a touch cooler and lower far from
      it) and deeper, cooler blue-grey shadow planes with a slight top-to-bottom gradient (deeper
      violet-grey high up under the anvil, lighter reflected blue low down);
    * the backlit crown: a narrow band just inside the sun-facing silhouette near the sun is held a
      notch down so the silver/gold lining reads against it."""
    u = W / 1920.0
    Hc = base_y - top_y
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    rgb = np.ascontiguousarray(rgba[..., :3])
    h, w = A.shape
    V = rgb.mean(-1)
    # --- one-sided softening
    Vi = V * A + 0.9 * (1 - A)
    Vb = cv2.GaussianBlur(Vi, (0, 0), 5 * u)
    gx = cv2.Sobel(Vb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Vb, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    kd = np.asarray(key_dir, np.float32)
    kd = kd / float(np.hypot(*kd))
    cosk = (gx * kd[0] + gy * kd[1]) / gl          # > 0: brighter toward the sun
    soft = K._ss(0.2, 0.7, cosk) * K._ss(0.004, 0.02, gl)
    soft = cv2.GaussianBlur(soft, (0, 0), 3 * u)
    Ae = cv2.erode((A > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    soft *= cv2.GaussianBlur(Ae, (0, 0), 2 * u)
    sig = 4.5 * u
    Ab = cv2.GaussianBlur(A, (0, 0), sig) + 1e-4
    rb = cv2.GaussianBlur(rgb * A[..., None], (0, 0), sig) / Ab[..., None]
    rgb = rgb * (1 - soft[..., None]) + rb * soft[..., None]
    V = rgb.mean(-1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    yn = np.clip((ys - top_y) / Hc, 0, 1)
    # --- lit faces: brilliant warm white, hottest toward the sun
    lit = K._ss(0.8, 0.9, V)[..., None]
    warm = np.exp(-dsun / 0.55)[..., None]
    lit_col = rgb * (1.0 + 0.035 * warm) * np.array([1.0, 0.99, 0.965], np.float32) ** warm         * (1 - 0.03 * yn[..., None])
    rgb = rgb * (1 - lit) + lit_col * lit
    # --- shadow planes: deeper + cooler, violet-grey high, reflected blue low
    sh = K._ss(0.78, 0.6, V)[..., None]
    hi_c = np.array([0.86, 0.85, 0.97], np.float32)
    lo_c = np.array([0.92, 0.97, 1.03], np.float32)
    g = yn[..., None]
    sh_col = rgb * (hi_c * (1 - g) + lo_c * g) * (0.9 + 0.08 * g)
    rgb = rgb * (1 - sh) + sh_col * sh
    # --- backlit crown: hold the lit value down just inside the sunward silhouette near the sun
    dT = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    Abl = cv2.GaussianBlur(A, (0, 0), 6 * u)
    nx = -cv2.Sobel(Abl, cv2.CV_32F, 1, 0, ksize=3)
    ny = -cv2.Sobel(Abl, cv2.CV_32F, 0, 1, ksize=3)
    nl = np.sqrt(nx * nx + ny * ny) + 1e-6
    face = K._ss(0.2, 0.8, (nx * kd[0] + ny * kd[1]) / nl)
    face = cv2.GaussianBlur(face * (nl > 1e-3), (0, 0), 14 * u)
    band = K._ss(2.5 * u, 7 * u, dT) * np.exp(-dT / (30 * u)) * np.clip(face * 2.0, 0, 1)
    band *= (0.45 + 0.55 * np.exp(-dsun / 0.4)) * (1 - K._ss(0.45, 0.75, yn))
    rgb = rgb * (1 - 0.2 * band[..., None] * np.array([1.0, 0.97, 0.86], np.float32))
    return np.concatenate([rgb, rgba[..., 3:4]], -1).astype(np.float32)


def top_edge(alpha, x, thr=0.5):
    col = alpha[:, int(np.clip(x, 0, alpha.shape[1] - 1))]
    nz = np.nonzero(col > thr)[0]
    return int(nz[0]) if len(nz) else None


def tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8'):
    """Paint the tower on a (ph, pw) straight-alpha plate; cx / base_y: foot centre, Hc: height to the
    anvil top. Returns (rgba, sun) with the sun seated on the anvil's upper-left crest."""
    u = W / 1920.0
    WF = 0.82                       # width factor: a tall, narrow rising column
    rng = np.random.default_rng(seed)
    by = base_y
    top = by - Hc
    # key light: high up-left, a little in front of the tower (the real sun is behind the crest)
    key = (cx - 0.9 * Hc, by - 1.9 * Hc)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=0.4)
    g = (cx + 0.02 * Hc, by - 0.5 * Hc, 0.42 * Hc, 0.62 * Hc)
    common = dict(group=g, form=0.25, term_w=0.08, wrap=0.0, brush=0.0, lit_step=0.3, term=0.45, hot=1.0,
                  sky=0.16, rim=0.0, rim_interior=0.0, split=0.38, scallop=1.15, term_noise=0.14,
                  levels=LEV_BIG, concave=0.7, clump=0.5, lit_bias=0.0)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        if not isinstance(env, np.ndarray):
            env = dict(env)
            env['cx'] = cx + (env['cx'] - cx) * WF
            env['width'] = env['width'] * WF
        poly = env if isinstance(env, np.ndarray) else K.envelope(rng=rng, **env)
        return P.shape(poly, rng=rng, **k)

    shade_low = dict(pal=PAL_BASE, light=0.6, lit_bias=-0.02, form=0.4, cast=0.35, wrap=0.3, sky=0.14,
                     inner=1, inner_val=0.55, inner_size=(0.35, 0.6), shade_top=0.08, valley_dark=0.06,
                     refl=0.6, bounce=0.4)
    # ---- anvil: grows out of the crown; blunt, rounded upwind end on the LEFT (the sun sits behind
    # its crest), long wedge downwind to the right that tears into fibres
    xa = cx + 0.02 * Hc
    long_, short_ = 0.6 * Hc, 0.24 * Hc
    ap = K.anvil_poly(xa, top + 0.03 * Hc, short_, long_, 0.16 * Hc, rng, dome=0.5, rise=0.35, tip=0.08,
                      sag=0.55)
    n_up = len(ap) // 2
    xx = ap[n_up:, 0]
    ap[n_up:, 1] += 0.014 * Hc * np.sin((xx - xa) / (0.08 * Hc) + rng.uniform(0, 6.28)) * \
        np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
    mass(ap, pal=PAL_ANVIL, levels=((0.06, 0.13, 0.6, 2.0, 4.0, 0.25, 0.5), (0.02, 0.045, 0.45, 1.4, 3.0)),
         size=0.2 * Hc, firm=0.7, soften=0.5, base_dark=0.6, side_scale=0.45, down_cut=0.05, clump=0.6,
         top_bias=1.8, inner=0, fray=(xa + 0.3 * long_, xa + 0.95 * long_, 1.25, 0.4 * long_), refl=0.55,
         wrap=0.0, bounce=0.25, shade_top=0.0, form=0.25, mass_r=0.45, split=0.1, scallop=0.6, sky=0.0,
         lit_bias=0.16, cast=0.0, group=(xa + 0.1 * long_, top + 0.1 * Hc, 0.6 * long_, 0.16 * Hc))
    # ---- crown heads punching up into the anvil's underside (staggered heights, no flat cap)
    mass(dict(cx=cx + 0.2 * Hc, base_y=by - 0.7 * Hc, width=0.3 * Hc, height=0.16 * Hc, power=2.2, lump=0.12,
              base_round=0.5, base_wave=0.06, skew=0.2), size=0.17 * Hc, cast=0.3, side_scale=0.6, form=0.4,
         lit_bias=0.02)
    mass(dict(cx=cx - 0.12 * Hc, base_y=by - 0.72 * Hc, width=0.36 * Hc, height=0.17 * Hc, power=2.2,
              lump=0.12, base_round=0.5, base_wave=0.06, skew=-0.2), size=0.19 * Hc, cast=0.3,
         side_scale=0.62, form=0.4, lit_bias=0.12)
    # ---- the rising column: one stout body from the base up into the crown, leaning right
    mass(dict(cx=cx + 0.03 * Hc, base_y=by - 0.22 * Hc, width=0.56 * Hc, height=0.58 * Hc, power=2.5,
              lump=0.08, lean=0.1, base_round=0.3, base_wave=0.05), size=0.34 * Hc, cast=0.3, side_scale=0.72,
         inner=1, inner_val=0.55)
    # ---- right flank: 2-3 big lobes bulging out on the shadow side, staggered down the column
    mass(dict(cx=cx + 0.3 * Hc, base_y=by - 0.52 * Hc, width=0.32 * Hc, height=0.2 * Hc, power=2.3, lump=0.1,
              skew=0.2, base_round=0.35, base_wave=0.08), size=0.2 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         wrap=0.0, lit_bias=0.14)
    mass(dict(cx=cx + 0.42 * Hc, base_y=by - 0.3 * Hc, width=0.34 * Hc, height=0.24 * Hc, power=2.3, lump=0.1,
              lean=0.04, skew=0.25, base_round=0.35, base_wave=0.08), size=0.22 * Hc, cast=0.35,
         side_scale=0.68, form=0.4, light=0.95, wrap=0.0, inner=1, inner_val=0.55, lit_bias=0.2)
    # ---- sun-side (left) lobes, lit-on-lit: crisp overlapping-mass edges inside the light
    mass(dict(cx=cx - 0.2 * Hc, base_y=by - 0.52 * Hc, width=0.3 * Hc, height=0.19 * Hc, power=2.3, lump=0.1,
              skew=-0.2, base_round=0.35, base_wave=0.08), size=0.19 * Hc, cast=0.25, side_scale=0.6,
         form=0.4, lit_bias=0.06)
    mass(dict(cx=cx - 0.02 * Hc, base_y=by - 0.46 * Hc, width=0.26 * Hc, height=0.15 * Hc, power=2.2, lump=0.1,
              skew=-0.1, base_round=0.3, base_wave=0.08), size=0.15 * Hc, cast=0.25, side_scale=0.6, form=0.35,
         lit_bias=0.04)
    mass(dict(cx=cx - 0.27 * Hc, base_y=by - 0.3 * Hc, width=0.32 * Hc, height=0.21 * Hc, power=2.3, lump=0.1,
              lean=-0.03, skew=-0.25, base_round=0.4, base_wave=0.08), size=0.2 * Hc, cast=0.3,
         side_scale=0.65, form=0.4, lit_bias=0.05)
    mass(dict(cx=cx + 0.14 * Hc, base_y=by - 0.28 * Hc, width=0.28 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=0.1, base_round=0.35, base_wave=0.08), size=0.17 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         lit_bias=0.16)
    # ---- low left shoulder, tearing into wisps on its far side
    mass(dict(cx=cx - 0.44 * Hc, base_y=by - 0.04 * Hc, width=0.3 * Hc, height=0.15 * Hc, power=2.3, lump=0.12,
              lean=-0.05, skew=-0.2), size=0.2 * Hc, cast=0.35, side_scale=0.7, form=0.4, inner=1, light=0.95,
         lit_bias=0.1, fray=(cx - 0.45 * Hc, cx - 0.66 * Hc, 0.9, 0.1 * Hc),
         group=(cx - 0.46 * Hc, by - 0.16 * Hc, 0.22 * Hc, 0.2 * Hc))
    for (fx_, fy_, fw_, fh_) in ((-0.6, -0.2, 0.08, 0.03),):
        mass(dict(cx=cx + fx_ * Hc, base_y=by + fy_ * Hc, width=fw_ * Hc, height=fh_ * Hc, power=2.2, lump=0.15),
             size=0.06 * Hc, form=0.3, light=0.85, alpha=0.7,
             fray=(cx + (fx_ + 0.03 * np.sign(fx_)) * Hc, cx + (fx_ + 0.07 * np.sign(fx_)) * Hc, 1.0, 0.05 * Hc),
             group=(cx + fx_ * Hc, by + (fy_ - 0.02) * Hc, 0.08 * Hc, 0.05 * Hc), inner=0, wrap=0.0)
    # ---- lower body: big overlapping masses of varied size, uneven heights
    mass(dict(cx=cx - 0.08 * Hc, base_y=by - 0.02 * Hc, width=0.6 * Hc, height=0.24 * Hc, power=2.4, lump=0.1,
              skew=0.2), size=0.26 * Hc, **shade_low)
    mass(dict(cx=cx + 0.36 * Hc, base_y=by + 0.02 * Hc, width=0.42 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=-0.2), size=0.18 * Hc, **dict(shade_low, light=0.5, lit_bias=-0.04))
    mass(dict(cx=cx - 0.3 * Hc, base_y=by + 0.04 * Hc, width=0.38 * Hc, height=0.12 * Hc, power=2.3, lump=0.1,
              skew=0.2), size=0.13 * Hc, **dict(shade_low, inner=0, light=0.7))
    # torn wispy underside, melting into the horizon haze
    P.wisps(cx - 0.66 * Hc, cx + 0.56 * Hc, by + 0.03 * Hc, 0.13 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 11,
            amount=0.45, erode=1.5, base_haze=0.65, haze_color=haze_col)
    # the base is torn apart again just above the rooftops' horizon line so the tower dissolves into the
    # haze there instead of standing on a clean lavender foot
    P.wisps(cx - 0.62 * Hc, cx + 0.56 * Hc, by - 0.19 * Hc, 0.07 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 13,
            amount=0.5, erode=1.1, base_haze=0.55, haze_color=haze_col)
    tipx = xa + long_
    P.fibres(tipx - 0.35 * long_, tipx + 0.45 * long_, top + 0.11 * Hc, 0.03 * Hc, pal=PAL_ANVIL, seed=seed + 21,
             amount=0.5, lit=0.85)
    P.fibres(tipx - 0.2 * long_, tipx + 0.3 * long_, top + 0.16 * Hc, 0.018 * Hc, pal=PAL_ANVIL, seed=seed + 23,
             amount=0.32, lit=0.6)
    rgba = P.rgba()
    # de-stair the thresholded value edges and remove the faint interior mottle
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.035, 3.0 * u + 0.5)
    rgba = np.concatenate([rgb, rgba[..., 3:4]], -1)
    # seat the sun on the anvil's upper-left crest: the highest point of the blunt upwind end
    best = None
    for x in np.linspace(xa - 0.3 * Hc, xa - 0.08 * Hc, 40):
        ey = top_edge(rgba[..., 3], x)
        if ey is not None and (best is None or ey < best[1] - 0.004 * Hc):
            best = (float(x), ey)
    sun = (best[0], best[1] + 0.004 * Hc) if best else (xa - 0.2 * Hc, top)
    rgba = repaint(rgba, sun, (-0.72, -0.7), W, top, by)
    rgba = sun_rim(rgba, sun, (-0.72, -0.7), W, top, by, strength=1.0)
    # warm translucent glow just inside the crest nearest the sun (light through the thin edge)
    ys, xs = np.mgrid[0:ph, 0:pw].astype(np.float32)
    A = rgba[..., 3]
    dT = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    gk = np.exp(-dT / (0.03 * Hc)) * np.exp(-dsun / 0.16) * A
    rgba[..., :3] += np.array([0.3, 0.22, 0.1], np.float32) * gk[..., None]
    # aerial perspective: the lower third loses contrast + saturation and fades to the pale haze
    yy = ys[:, :1]
    k = (K._ss(by - 0.5 * Hc, by + 0.04 * Hc, yy) ** 1.1)[..., None]
    rgb = rgba[..., :3]
    lum = rgb.mean(-1, keepdims=True)
    rgb = rgb * (1 - 0.45 * k) + lum * 0.45 * k
    mu = 0.82
    rgb = rgb * (1 - 0.35 * k) + (rgb * 0.0 + (lum - mu) * 0.4 + mu) * 0.35 * k   # squash contrast
    rgb = rgb * (1 - 0.62 * k) + K._c(haze_col) * (0.62 * k)
    # close to the horizon the foot melts into a pale, almost sky-coloured haze
    kh = (K._ss(by - 0.45 * Hc, by - 0.2 * Hc, yy) ** 1.2)[..., None]
    rgb = rgb * (1 - 0.6 * kh) + K._c(haze_col) * (0.6 * kh)
    al = rgba[..., 3:4] * (1 - 0.3 * k * K._ss(by - 0.18 * Hc, by + 0.06 * Hc, yy)[..., None]) * (1 - 0.35 * kh)
    rgba = np.concatenate([rgb, al], -1).astype(np.float32)
    return rgba, sun
