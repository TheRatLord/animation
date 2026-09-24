"""s01 helper: the hero cumulonimbus (painted with lib/clouds2) + distant horizon cumulus.

Layout: a broad hazy base, a stout column that WIDENS upward into a broad cauliflower crown (the anvil
beginning to spread), a shorter secondary tower on the shadow (right) side and a low left shoulder that
tears into wisps and fractus. One tower-wide light from the upper left (the sun sits just behind the
crown's top edge). Values are big flat painted masses: a near-white core facing the sun, warm cream,
a cool lavender-grey mid step, crisp lit-on-lit overlap edges; the shadow is a desaturated blue-grey
(greyer / warmer low down) broken by a few big overlapping masses with lighter reflected-light lips.
The silver/gold lining sits only on the crown and the sun-facing upper-left lobes; shadow-side edges
meet the sky softly. Aerial perspective dissolves the lowest quarter of the tower into the horizon."""
import numpy as np
import cv2

from lib import clouds2 as K


# lit side: near-white core toward the sun -> warm cream -> a cool lavender-grey mid step; shadow: a
# desaturated blue-grey (#8C9BC8 family), greyer / warmer toward the base (bounce from the ground haze)
PAL = dict(hi=(0.995, 0.985, 0.965), lit='#f3e9e0', lit_lo='#dad7e2', mid='#c8bfcf', shade='#93a0cb',
           deep='#8190c0', refl='#adb8d8', bounce='#c6c0c8', rim=(1.55, 1.46, 1.22), haze='#c9e2f4',
           edge_dark=0.0)
PAL_BASE = dict(PAL, lit='#e9e2e0', lit_lo='#cfcfda', hi=(0.96, 0.95, 0.94), shade='#97a2c6', deep='#8994bc',
                refl='#b0b9d4', bounce='#c9c3c8')
# cauliflower levels: a few BIG heads, mid florets riding on them, a sprinkling of small ones (varied
# scale along the edge, never one uniform scallop size)
LEV_BIG = ((0.14, 0.4, 0.95, 1.2, 2.4, 0.12, 0.45), (0.05, 0.13, 0.85, 1.0, 2.2, 0.3, 0.6),
           (0.02, 0.045, 0.6, 1.2, 3.0), (0.012, 0.02, 0.2, 1.6, 4.0))


def _top_edge(alpha, x, thr=0.5):
    """First row where the plate becomes opaque in column x (or None)."""
    col = alpha[:, int(np.clip(x, 0, alpha.shape[1] - 1))]
    nz = np.nonzero(col > thr)[0]
    return int(nz[0]) if len(nz) else None


def _sun_rim(rgba, sun, key_dir, W, top_y, base_y, strength=1.0):
    """Silver/gold lining ONLY where the outer silhouette faces the light: the crown near the sun
    (backlit: edges within reach of the sun glow) and the up/left-facing lobes of the upper, sun-facing
    flank. Shadow-side and lower edges meet the sky with a soft cool edge and no rim."""
    u = W / 1920.0
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    h, w = A.shape
    Ab = cv2.GaussianBlur(A, (0, 0), 5 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl                                   # outward normal
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    Hc = base_y - top_y
    d = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    near = np.exp(-(d / 0.22) ** 2)                               # backlit zone round the sun
    kd = np.asarray(key_dir, np.float32)
    kd = kd / float(np.hypot(*kd))
    face = np.clip(nx * kd[0] + ny * kd[1], 0, 1) ** 1.5
    vz = 1 - K._ss(top_y + 0.3 * Hc, top_y + 0.62 * Hc, ys)      # upper part of the tower only
    k_face = face * vz * (0.55 + 0.45 * np.exp(-d / 0.5))
    k = np.clip(np.maximum(k_face, near * np.clip(0.35 + 0.65 * (-ny), 0, 1)), 0, 1)
    rb = max(int(round(1.6 * u)), 1)
    core = np.clip(A - cv2.erode(A, np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)), 0, 1)
    rb2 = max(int(round(3.2 * u)), 1)
    wide = np.clip(A - cv2.erode(A, np.ones((2 * rb2 + 1, 2 * rb2 + 1), np.uint8)), 0, 1)
    wide = cv2.GaussianBlur(wide, (0, 0), 1.0 * u)
    ring = np.maximum(core, wide * (0.35 + 0.5 * near))
    kk = np.clip(ring * k * strength, 0, 1)
    silver = np.array([1.35, 1.36, 1.36], np.float32)
    gold = np.array([1.7, 1.46, 1.08], np.float32)
    col = silver * (1 - near[..., None]) + gold * near[..., None]
    rgb = rgba[..., :3] * (1 - kk[..., None]) + col * kk[..., None]
    hk = cv2.GaussianBlur(kk, (0, 0), 2.5 * u) * 0.7 + cv2.GaussianBlur(kk, (0, 0), 8 * u) * 0.5
    hk = np.clip(hk * (1 - A) * (0.3 + 0.9 * near), 0, 1)
    a = np.clip(A + hk * 0.45, 0, 1)
    rgb = (rgb * A[..., None] + col * hk[..., None]) / np.maximum(a, 1e-4)[..., None]
    rgb = np.where(a[..., None] > 1e-4, rgb, rgba[..., :3])
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32)


def tower(pw, ph, x0, base_y, Hc, W, sun_guess, seed=12, sun_inset=0.012):
    """Paint the tower; return (rgba, sun_p) with sun_p re-seated on the crown silhouette (slightly
    inside it, so the disc is hidden and only its glow, rays and flare spill past the edge)."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    key = (x0 - 0.7 * Hc, base_y - 1.3 * Hc)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=0.3)
    cx, by = x0, base_y
    g = (cx + 0.02 * Hc, by - 0.52 * Hc, 0.48 * Hc, 0.62 * Hc)
    common = dict(group=g, form=0.55, term_w=0.1, wrap=0.25, brush=0.0, lit_step=0.3, term=0.55, hot=1.0,
                  sky=0.12, rim=0.0, rim_interior=0.0, split=0.26, scallop=1.1, term_noise=0.12, levels=LEV_BIG,
                  concave=0.7, clump=0.5)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        poly = env if isinstance(env, np.ndarray) else K.envelope(rng=rng, **env)
        return P.shape(poly, rng=rng, **k)

    shade_low = dict(pal=PAL_BASE, light=0.45, lit_bias=-0.12, form=0.4, cast=0.5, wrap=0.15, sky=0.1,
                     inner=1, inner_val=0.42, inner_size=(0.35, 0.6), shade_top=0.15, valley_dark=0.12,
                     refl=0.5, bounce=0.35)
    # ---- back: broad base filler (tower's own shade)
    mass(dict(cx=cx + 0.06 * Hc, base_y=by + 0.02 * Hc, width=1.1 * Hc, height=0.42 * Hc, power=2.5, lump=0.08),
         size=0.3 * Hc, **dict(shade_low, inner=0))
    # ---- secondary tower on the shadow side (right), shorter
    mass(dict(cx=cx + 0.36 * Hc, base_y=by - 0.2 * Hc, width=0.34 * Hc, height=0.42 * Hc, power=2.6, lump=0.1,
              lean=0.06, skew=0.2), size=0.3 * Hc, cast=0.35, side_scale=0.72, light=0.85, inner=1,
         inner_val=0.5, group=(cx + 0.36 * Hc, by - 0.42 * Hc, 0.2 * Hc, 0.26 * Hc))
    # ---- crown: broad cap wider than the column (widens upward), sun behind its upper edge
    mass(dict(cx=cx + 0.04 * Hc, base_y=by - 0.68 * Hc, width=0.8 * Hc, height=0.3 * Hc, power=2.3, lump=0.1,
              base_round=0.5, base_wave=0.06, skew=0.1), size=0.28 * Hc, cast=0.3, side_scale=0.62, down_cut=0.2, lit_bias=0.1,
         group=(cx + 0.05 * Hc, by - 0.8 * Hc, 0.36 * Hc, 0.22 * Hc), form=0.5)
    # crown heads (left big / right smaller) punching up
    mass(dict(cx=cx - 0.17 * Hc, base_y=by - 0.79 * Hc, width=0.36 * Hc, height=0.19 * Hc, power=2.2, lump=0.12,
              base_round=0.45, base_wave=0.06), size=0.2 * Hc, cast=0.45, side_scale=0.6, form=0.45)
    mass(dict(cx=cx + 0.2 * Hc, base_y=by - 0.82 * Hc, width=0.26 * Hc, height=0.16 * Hc, power=2.2, lump=0.12,
              base_round=0.45, base_wave=0.06, skew=0.2), size=0.17 * Hc, cast=0.45, side_scale=0.6, form=0.45)
    # ---- main column: stout, slightly leaning right, meeting the crown's underside
    mass(dict(cx=cx - 0.02 * Hc, base_y=by - 0.26 * Hc, width=0.5 * Hc, height=0.5 * Hc, power=2.6, lump=0.09,
              lean=0.05, base_round=0.3, base_wave=0.05), size=0.34 * Hc, cast=0.4, side_scale=0.72, inner=1, lit_bias=0.08)
    # lit-on-lit lobes on the column's sunward face (crisp overlapping-mass edges in the light)
    mass(dict(cx=cx - 0.13 * Hc, base_y=by - 0.52 * Hc, width=0.3 * Hc, height=0.18 * Hc, power=2.3, lump=0.1,
              skew=-0.2, base_round=0.3, base_wave=0.08), size=0.18 * Hc, cast=0.3, side_scale=0.6, form=0.4, light=1.0, lit_bias=0.1)
    # mid-right lobe stacked on the column's shadowed flank
    mass(dict(cx=cx + 0.2 * Hc, base_y=by - 0.4 * Hc, width=0.26 * Hc, height=0.2 * Hc, power=2.4, lump=0.1,
              skew=-0.2, base_round=0.5, base_wave=0.08), size=0.2 * Hc, cast=0.55, side_scale=0.7, form=0.4, light=0.85, inner=1, inner_val=0.5)
    # ---- left shoulder: low, lit, tearing into wisps on its far side
    mass(dict(cx=cx - 0.4 * Hc, base_y=by - 0.1 * Hc, width=0.36 * Hc, height=0.28 * Hc, power=2.3, lump=0.12,
              lean=-0.05, skew=-0.2), size=0.24 * Hc, cast=0.45, side_scale=0.7, form=0.45, inner=1,
         fray=(cx - 0.45 * Hc, cx - 0.62 * Hc, 0.9, 0.12 * Hc),
         group=(cx - 0.4 * Hc, by - 0.22 * Hc, 0.22 * Hc, 0.22 * Hc))
    # small detached fragments beside the shoulder (fractus)
    for (fx_, fy_, fw_, fh_) in ((-0.66, -0.2, 0.12, 0.05), (-0.6, -0.3, 0.08, 0.035), (-0.72, -0.12, 0.1, 0.04)):
        mass(dict(cx=cx + fx_ * Hc, base_y=by + fy_ * Hc, width=fw_ * Hc, height=fh_ * Hc, power=2.2, lump=0.15),
             size=0.06 * Hc, form=0.3, light=0.8, alpha=0.8,
             fray=(cx + (fx_ + 0.03) * Hc, cx + (fx_ - 0.06) * Hc, 1.0, 0.05 * Hc),
             group=(cx + fx_ * Hc, by + (fy_ - 0.02) * Hc, 0.08 * Hc, 0.05 * Hc), inner=0, wrap=0.0)
    # ---- lower body in the tower's own shade: a few big overlapping masses of varied size
    mass(dict(cx=cx + 0.08 * Hc, base_y=by - 0.1 * Hc, width=0.56 * Hc, height=0.28 * Hc, power=2.4, lump=0.1,
              skew=0.2), size=0.3 * Hc, **shade_low)
    mass(dict(cx=cx + 0.42 * Hc, base_y=by - 0.04 * Hc, width=0.34 * Hc, height=0.22 * Hc, power=2.3, lump=0.1,
              skew=0.3), size=0.22 * Hc, **shade_low)
    mass(dict(cx=cx - 0.14 * Hc, base_y=by + 0.02 * Hc, width=0.4 * Hc, height=0.16 * Hc, power=2.3, lump=0.1,
              skew=-0.2), size=0.15 * Hc, **dict(shade_low, inner=0, light=0.6))
    # torn, wispy underside + torn shoulder base
    P.wisps(cx - 0.7 * Hc, cx + 0.66 * Hc, by + 0.02 * Hc, 0.1 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 11,
            amount=0.55, erode=1.2, base_haze=0.45)
    P.wisps(cx - 0.62 * Hc, cx - 0.3 * Hc, by - 0.12 * Hc, 0.05 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 17,
            amount=0.4, erode=0.5, base_haze=0.0)
    rgba = P.rgba()
    # de-stair the painted value edges (median removes the small teeth of thresholded fields while
    # keeping the edges crisp)
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.03, 2.0 * u + 0.5)
    rgba = np.concatenate([rgb, rgba[..., 3:4]], -1)

    # seat the sun on the silhouette (behind the crown's upper edge)
    sx = sun_guess[0]
    ey = _top_edge(rgba[..., 3], sx)
    sun_p = (sx, (ey + sun_inset * W) if ey is not None else sun_guess[1])
    top_y = by - Hc
    rgba = _sun_rim(rgba, sun_p, (-0.55, -1.0), W, top_y, by)

    # aerial perspective: the bottom ~quarter of the visible tower dissolves toward the pale horizon
    # (lower contrast, lighter, cooler) - it stands tens of km away behind the town
    ys = np.arange(ph, dtype=np.float32)[:, None]
    k = (K._ss(by - 0.42 * Hc, by - 0.1 * Hc, ys) ** 1.3)[..., None]
    rgb = rgba[..., :3]
    lum = rgb.mean(-1, keepdims=True)
    rgb = rgb * (1 - 0.35 * k) + lum * 0.35 * k                     # contrast / saturation loss
    haze_col = K._c('#d2e8f6')
    rgb = rgb * (1 - 0.55 * k) + haze_col * (0.55 * k)
    rgba = np.concatenate([rgb, rgba[..., 3:4] * (1 - 0.12 * k)], -1).astype(np.float32)
    return rgba, sun_p


def bank(pw, ph, y, W, sun_p, seed=41):
    return K.horizon_bank_plate(pw, ph, y, 0.07 * W * 9 / 16, sun=sun_p, preset='noon', seed=seed, rows=4,
                                haze=0.3, haze_color='#d4ecf8', shrink=0.4, layer_haze=0.12, unit=W / 1920.0)
