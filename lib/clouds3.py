"""clouds3 - painted Shinkai-style clouds, round 3 final (volumetric lobe painter + legacy engines).

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
        [far_plate, near_plate] for parallax, with split=(d1, d2, ...) returns [far, ..., nearest]
        (each band is painted from its own + farther lobes, so parallax never uncovers holes). Put
        side-view towers (cumulonimbus_plate + sea_ground_y) between the bands for correct occlusion.
    sea_ground(w, h, horizon_y, x, dist, fov=55, cam_height=1) -> (base_y, px_per_unit)
        placement of a side-view tower plate standing on the deck at distance `dist`
    cirrus_plate(w, h, preset, seed, region=(0.05, 0.4), angle=-8, density=0.5, opacity=0.6)
    cloudlets_plate(w, h, preset, seed, region=(0.05, 0.45), density=0.5, size=0.012, angle=-12,
                    stretch=2.2, opacity=0.85, sun_dir=(0.6, -0.8), patch=0.35, persp=1.8, rows=1.0,
                    avoid=[(x, y, rx, ry)], soft=1.0, ripple=1.0, body=None, lit=None)
        mackerel sky: rippled rows of lumpy flecks along a flow direction, perspective-scaled, gathered
        in patches, lit sun-side edge + soft trailing underside, thinned inside `avoid` ellipses
    stratus_plate(w, h, y, thick, preset, seed, x_range=(-0.05, 1.05), count=6, length=(0.25, 0.7),
                  top_color, body_color, under_color, opacity=0.9, streak=1.0, sun_x=None)
        flat horizon strata: lens-shaped bars, lit scalloped top, flat darker lost base, brush streaks
    haze_plate(w, h, y0, y1, color, amount=0.6, y2=None)
    dissolve_base(plate, cx, half_w, base_y, fade_h, seed)   torn, lost base (in place)
Painting controls (**paint, forwarded to render_lobes; defaults tuned for Shinkai day clouds)
    form (0.55) big-form normal weight, head (0.6) head-normal weight, micro (0.06) floret normals,
    wrap (0.35), density (2.2) self-shadow, shadow_soft (0.35), contrast (0.45) / pivot (0.3) lit-vs-
    shadow simplification, bands (0.6) painted value steps, band_noise, warp (brushy boundaries),
    edge (1.0) crisp lit lobe-top / crest bands, crest_focus=(x, width, floor) (crests hottest near the
    sun axis), limb (0.25) white kept for the sunward silhouette, rim / rim_px lining, glow backlit
    forward scatter, dimple (1.0) fills AO-like pits, paint (1.0) Kuwahara brush pass, dab_amt (0.025),
    lost (1.0) soft shadow-side silhouettes, base_soft, haze / fog aerial perspective, keep (mask),
    base_shade (0) darkens the body toward its base (lit crown, shaded base), crest_fade=(y0, y1, floor)
    (plate rows: crest / rim highlights fade from full at y0 to `floor` at y1), rim_wrap (0.1) how far the
    lining wraps past the sun-facing silhouette (linings are only painted on sun-facing contours).
FINAL-ROUND ADDITIONS (use these for new shots)
    style='flat' (render_lobes / every side-view plate, via **paint) + flat=dict(...): the painted
        flat-plane painter. Light is taken from the big form + head normals x cast self-shadow, cleaned
        of pits / specks (islands), then stepped into 3 flat planes (shade / mid / lit) with crisp
        brush-warped borders; no per-lobe sphere shading, no AO dimples. flat keys (FLAT_DEFAULTS):
        t_lo, t_hi, step (plane thresholds / border softness), noise, curl, warp (border wobble),
        islands (pit / speck removal, x lobe size), smooth, grad (gradient inside planes), lit / lit_in /
        lit_span, mid, shade, sky, hi, lit_lo, deep_col (colour overrides), lit_warm + warm_dir (warm
        cream on the sun flank), sky_fill, bounce, deep, limb / limb_px (brightest white hugging the
        sunward silhouette), crest_lit / crest_shade / crest_px (crisp lit outline where one HEAD
        overlaps another, smeared into the cloud), jump0 / jump1, head_lines, sep / sep_px (soft shade
        band just below each head outline - separates stacked heads), florets / floret_cut /
        floret_dark / floret_range (stepped cauliflower crescents inside the lit masses), brush /
        brush_angle / brush_tint (brush texture + hue shift).
    cloud_lobes / shape=dict(...): recede (upper heads step back -> tower leans into depth, no
        stacked-snowman read), sub_head (florets bigger than this x half-width start their own head).
    sky_streets_plate(w, h, horizon_y, preset, seed, region, fov, vp_x, size, density, patch, streets,
                      spacing, band_stretch, opacity, sun_dir, warm, avoid, lit, body, warm_col, veil,
                      dab_angle, lumps, aspect, min_px)
        mackerel sky on a perspective sky plane: streets converging on vp_x, shoals stretched along
        them, ragged dry-brush dabs fading by alpha, larger overhead, veil under each shoal.
    deck_rows_plate(w, h, horizon_y, sun_x, preset, seed, fov, z_near, z_far, row_step, head_w, head_h,
                    rim_px, rim_focus, rim_floor, rim_near, top_col, body_col, deep_col, rim_col,
                    far_col, gap, gap_col, haze, near_dark, glow, top_band, curl, brush, texture,
                    split=(d1, d2, ...))
        sea of clouds as painted receding rows (crisp gold crest rims gathered toward the sun column
        and horizon, flat violet bodies darkening toward the viewer, dark gaps, horizon haze);
        with split returns [far, ..., nearest] plates for parallax. Towers: cumulonimbus_plate at
        base_y = horizon_y + f / z, height = hgt * f / z (f = (w/2) / tan(fov/2)).
    sea_plate(w, h, horizon_y, sun_x, near, far, glitter, path_w, seed, haze) -> (plate, glit)
    sea_glitter(glit, t, W, H, cam, zoom, depth)   additive, coherent twinkle (no per-frame noise)
    power_lines_plate(w, h, pole_x, pole_top, seed, color, wires, sag, left_y, second)  silhouette
    shinkai_flare(W, H, lx, ly, ...)  one-shot flare (glow, 6-point star, prismatic ring, ghost chain,
        optional vertical bead line); SunFlare(W, H, tint, star, ring, chain, bloom, star_len,
        vertical).render(lx, ly, intensity, t, center) is the cached per-frame version (~0.05 s).
    surface_grow draws every per-candidate random number before filtering, so a plate looks the same
        at 960 and 1920 wide (resolution independent layout).
VOLUMETRIC PAINTER (final; used by scenes/_c3_X_noon.py and _c3_X_sea.py - prefer it for new shots)
    Clouds are 3D lobe clusters voxelised into an SDF grid (billow erosion), ray-marched with a real sun
    march (+ multiple-scattering octaves, dual HG phase), sky-ambient occlusion and a big-form Lambert term,
    then PAINTED: value -> warm/cool ramp (VRAMPS 'noon' / 'sunrise'), highlight roll-off, Kuwahara
    flattening into broad value masses, slow warm/cool colour drift, dry-brush strokes, crisp edges where
    lit / lost soft edges on the shadow side and base, and fine cauliflower florets on the lit silhouette.
    vol_cloud_plate(w, h, clouds=[dict(cx, base_y, width, height, seed, kind='tower'|'heap', lean, dz,
                    n_heads, r_rng, shrink, sizes, anvil=dict(left, right, thick, col_hw, neck))],
                    sun_dir=(0.75, -0.65), sun_z=-0.1 | L3=(x, y, z) light vector, eye_y, persp, max_dim=260,
                    render=dict(-> vol_render), paint=dict(-> paint_vol), florets=True|dict, haze, haze_col)
        -> straight RGBA plate. ~4-8 s per tower at 1080p (first run after an edit adds numba JIT time).
    head_lobes(...) tower / heap lobe layout: big flattened heads (tower-on-tower, shrinking upward, lower
        heads nearer), fine cauliflower grown on the exposed upper / sunward surface. anvil_lobes(...) flat-
        topped anvil sheet + flared neck (wwy_02). plume_lobes / tower_lobes: alternative layouts.
    vol_render(E, w, h, f, ppx, ppy, L, ...) -> fields (sun, amb, A, depth, lam); paint_vol(R, w, h, ramp,
        k_sun, k_amb, tone_k, lam_floor, crisp, lost, kuwa, tint_var, strokes, base_y, base_dark, backlit=None)
    edge_florets(P, sun_dir, r, density, ...) fine lobes on a plate's lit silhouette (in place).
    deck_fields(w, h, horizon_y, sun=(x, y), fov, cam_h, seed, towers=[(X, Z, radius, top)], bank, p1..p4,
        gap, stretch) -> ray-marched sea-of-clouds heightfield volume in perspective (fields sun, amb, A, Z,
        shadow; towers cast shadow wedges); paint_deck(R, w, h, horizon_y, sun_x, ...) -> RGB: navy-violet
        bodies, gold crest rims focused toward the sun / distance / light patches, horizon haze, airbrushed
        near banks. Keep the Z field for per-pixel parallax (see _c3_X_sea.Scene._warp_deck).
    sky_rows_plate(w, h, horizon_y, seed, fov, alt, angle, row_l, cell, cell_stretch, patch, patch_thr, cover,
        warp, lit, body, under, lit_screen=(dx, dy), veil, avoid, ...) -> mackerel / altocumulus streets on a
        perspective sky plane (large overhead, compressing into streaks at the horizon), lit edge + soft
        trailing side, gathered in patches.
    strata_plate(w, h, y, thick, seed, count, length, opacity, top, body, under) feathered stratus layers.
PAINTED-HEADS ENGINE (previous 2D engine, kept for compatibility; the demo scenes now use the VOLUMETRIC PAINTER)
    A cloud is an ordered list of 2D HEADS (back -> front), each a core ellipse + lumps + 3 scales of
    cauliflower bumps on its upper arc, painted like a painter stacks them. Light = big-form field of the
    union silhouette (normal x screen-space sun march, darker toward the base) + each head's own
    crust light (depth inside the head toward the sun -> terminator follows the scalloped outline).
    Stepped painted planes, warm/cool ramp, rim / glow only on sun-facing silhouettes, anisotropic
    edges (crisp lit / up side, lost down / shadow side and base), grey-closing removes AO-like pits.
    HeadSet().add(cx, cy, rx, ry, rng, k, base_y, top_y, haze, tone, aa, soft, rim, rim_px, warm, beta,
                  glow, light=(Lx, Ly, Lz) per-head light, bump, arc, lumps, sub, sub3, min_px)
    tower_heads(hs, rng, cx, base_y, height, width, lean, sun_dir, kind='cb'|'heap', sc, haze, children,
                k_levels, bump, rim, rim_px, glow, base_soft, tone, ...)     lays out one tower / heap
    anvil_heads(hs, rng, xt, top_y, left, right, thick, sc, flare, ...)     flat-topped anvil sheet
    add_fringe(hs, w, h, rng, sun_dir, r, density, sc, **head_kw)            fine florets on lit contour
    paint_heads(w, h, hs, sun_dir, sun_z, ramp='noon'|'sunrise'|dict, dens, floor, crust, crust_mix,
                poster, brush, kuwa, k_inside, soft_inside, edge_noise, close_px, rim_sil, rim_up,
                warm_split, base_y, base_dark, Vbig) -> RGBA plate
    cloud_tower_plate(w, h, cx, base_y, height, width, sun_dir, sun_z=0.95 (noon) / < 0 (backlit),
                      ramp, seed, lean, kind, anvil=dict(left, right, thick), fringe, haze, tower, fr, **paint)
    heaps_plate(w, h, [dict(cx, base_y, height, width, seed, haze, lean, kind)], sun_dir, ..., **paint)
    deck_plate(w, h, horizon_y, sun=(x, y), seed, fov, z_near, z_far, row_step, head_w, rim, rim_px,
               rim_focus, near_dark, haze, split=(d1, d2)) -> sea of clouds (list far..near with split)
    flecks_plate(w, h, horizon_y, seed, region, streams, angle, dab_angle, size, per_stream, width,
                 avoid, sun_dir, lit, body, under, veil, haze_col, dry)   mackerel / altocumulus streams
    mackerel_plate(...)            alternative cloudlets built from tiny heads
    utility_pole_plate(w, h, x, top, sun_dir) -> (RGBA, wire anchors); draw_wires(W, H, wires, t, sun)
        -> (RGBA layer, additive glint); iridescence_plate(alpha, center, radius, strength)
    Cost at 1080p: tower ~4-5 s, deck (3 bands) ~10 s, flecks ~2-3 s; demo scenes set up in ~22 s
    and render ~1.0-1.2 s / frame.
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
    'noon': dict(hi=(1.07, 1.03, 0.9), lit=(1.0, 0.965, 0.87), lit_lo=(0.86, 0.9, 0.94), mid=(0.7, 0.79, 0.92),
                 shade=(0.5, 0.66, 0.85), deep=(0.37, 0.5, 0.75), sky=(0.64, 0.78, 0.95),
                 bounce=(0.62, 0.69, 0.86), rim=(1.3, 1.26, 1.12), haze=(0.74, 0.86, 0.97)),
    'magic_hour': dict(hi=(1.06, 0.96, 0.8), lit=(1.0, 0.86, 0.66), lit_lo=(0.94, 0.72, 0.66), mid=(0.86, 0.6, 0.7),
                       shade=(0.6, 0.58, 0.8), deep=(0.36, 0.34, 0.62), sky=(0.62, 0.66, 0.9),
                       bounce=(0.9, 0.7, 0.66), rim=(1.4, 1.15, 0.8), haze=(0.94, 0.8, 0.76)),
    'sunset': dict(hi=(1.08, 0.86, 0.6), lit=(1.0, 0.68, 0.45), lit_lo=(0.94, 0.54, 0.54), mid=(0.84, 0.44, 0.6),
                   shade=(0.5, 0.42, 0.7), deep=(0.22, 0.18, 0.44), sky=(0.52, 0.5, 0.82),
                   bounce=(0.88, 0.52, 0.52), rim=(1.45, 1.05, 0.62), haze=(0.92, 0.62, 0.66)),
    'sunrise': dict(hi=(1.12, 0.92, 0.64), lit=(1.04, 0.76, 0.5), lit_lo=(0.92, 0.58, 0.58), mid=(0.7, 0.42, 0.6),
                    shade=(0.3, 0.29, 0.54), deep=(0.11, 0.11, 0.31), sky=(0.42, 0.41, 0.68),
                    bounce=(0.78, 0.5, 0.6), rim=(1.5, 1.1, 0.68), haze=(0.94, 0.7, 0.72)),
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
                 region=None, hier=0.0, sub_head=0.0):
    """Multi-scale cauliflower growth: for each radius in `sizes` (px, big -> small) sample points on the
    EXPOSED surface of the current lobe union and add a lobe there (centre sunk by `sink` x r).

    dens: coverage (lobes x r^2 / area); area: surface area estimate (px^2); up: world up (camera space);
    front: thinning of lobes on camera-facing surfaces (detail lives on the silhouette and tops);
    down_cut: no growth where the surface faces down more than this (flat, smooth bases);
    clump: 0..1 low-frequency 3D modulation of the density (dense cauliflower passages vs smooth broad
    ones); flat: vertical squash of new lobes; sun / sun_bias: extra growth on the sunward side;
    jitter: radius spread; region: optional callable(P (M, 3)) -> bool mask to restrict growth;
    hier: 0..1 scale hierarchy - finer levels are pushed off the camera-facing body onto the silhouette
    and the tops (big smooth masses inside, cauliflower only on the contour).
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
        li = sizes.index(rk) / max(len(sizes) - 1, 1)          # 0 = biggest level .. 1 = finest
        fexp = 1.5 / (1.0 + hier * 2.5 * li)                     # finer levels: thinned on the face, kept on the rim
        pr *= 1 - min(front + hier * 0.15 * li, 0.995) * np.clip(-nrm[:, 2], 0, 1) ** fexp
        if hier:
            pr *= 1 - hier * li * 0.6 * np.clip(-u, 0, 1)        # fine cauliflower lives on tops, not undersides
        pr *= np.where(nrm[:, 2] > 0.45, 0.0, 1.0)          # back side: invisible, skip
        if sun is not None:
            pr *= 1 + sun_bias * (nrm @ np.asarray(sun))
        if clump:
            f = np.sin(P @ K.T + ph).mean(1) * 2.0
            pr *= np.clip(1 - clump + clump * (0.5 + f), 0, 1.5)
        if region is not None:
            pr *= region(P)
        keep = rng.random(n) < pr * 0.8
        # every per-candidate random number is drawn BEFORE filtering, so a borderline filter decision
        # (float round-off at another resolution) never shifts the random stream (resolution independence)
        r_all = rk * rng.uniform(jitter[0], jitter[1], n)
        k_all = rng.uniform(sink[0], sink[1], n)
        fx_all = rng.uniform(1.0, 1.2, n)
        fy_all = rng.uniform(0.85, 1.0, n)
        P, nrm, idx = P[keep], nrm[keep], idx[keep]
        r_all, k_all, fx_all, fy_all = r_all[keep], k_all[keep], fx_all[keep], fy_all[keep]
        if len(P) == 0:
            continue
        # buried test against every other lobe (chunked)
        ok = _exposed(np.ascontiguousarray(P), np.ascontiguousarray(Cs), np.ascontiguousarray(Rs),
                      np.ascontiguousarray(idx.astype(np.int64)), 0.97)
        P, nrm, idx = P[ok], nrm[ok], idx[ok]
        m = len(P)
        if m == 0:
            continue
        r = r_all[ok]
        k = k_all[ok]
        cn = P - nrm * (r * k)[:, None]
        r3 = np.stack([r * fx_all[ok], r * flat * fy_all[ok], r], 1)
        Cs = np.concatenate([Cs, cn], 0)
        Rs = np.concatenate([Rs, r3], 0)
        n0 = len(Rs) - m
        own = np.arange(n0, n0 + m)
        Ro = np.concatenate([Ro, np.where(r >= sub_head, own, Ro[idx]) if sub_head else Ro[idx]], 0)
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


FLAT_DEFAULTS = dict(t_lo=0.22, t_hi=0.5, step=0.02, grad=0.18, noise=0.06, warp=0.3, crest_lit=0.9, crest_shade=0.3,
                     deep=0.3, sky_fill=0.6, bounce=0.35, mid_mix=0.5, head_lines=True, shade_lift=0.0, lit_warm=0.0)


def _paint_flat(hit, hitf, t0, od, Nb, Ns, L, pal, up, pn, band, sidc, E, lobe_px, ss, seed, density, wrap,
                shadow_soft, base_shade, edge, opt, rl=None):
    """Painted flat value planes (style='flat'): the light is taken from the BIG form only (inflated
    silhouette normal x cast self-shadow), then stepped into 3 flat planes - a cool shadow mass, a pale
    mid plane and a warm lit plane - with crisp, brush-warped borders and only a faint gradient inside
    each plane. Lobe / floret normals never shade the body (no clay balls, no AO dimples); the cauliflower
    reads through the silhouette and through thin crisp lit crest lines along head outlines."""
    o = dict(FLAT_DEFAULTS)
    o.update(opt or {})
    H2, W2 = hit.shape
    Lf = np.asarray(L, F32)
    lam = np.clip((Nb @ Lf + wrap) / (1 + wrap), 0, 1)
    vis_raw = np.exp(-od * density)
    ssig = max(shadow_soft * lobe_px * ss, 0.6)
    vis = np.clip(_blur(vis_raw * hitf, ssig) / (_blur(hitf, ssig) + 1e-4), 0, 1)
    if E.shape[1] > 12:
        dk = E[:, 12].astype(F32)[sidc] * hitf
        vis = vis * (1 - dk)
    Lb = (lam * vis).astype(F32)
    if _DEBUG is not None:
        _DEBUG.update(flam=lam, fvis=vis)
    flatb = (hit & (Ns @ np.asarray(pn, F32) > 0.97)).astype(F32)
    if flatb.any():
        Lb = Lb * (1 - 0.8 * _blur(flatb, 1.0))
    if base_shade:
        rows_ = np.nonzero(hit.any(1))[0]
        if len(rows_):
            yn_ = (np.arange(H2, dtype=F32) - rows_[0]) / max(rows_[-1] - rows_[0], 1)
            Lb = Lb * (1 - base_shade * _ss(0.35, 1.0, yn_))[:, None]
    if o.get('islands', 0.0):
        # painters never leave pits or specks: grey closing (fills small dark pits) + opening (removes
        # small bright specks) at a fraction of the lobe size
        kd = max(3, int(o['islands'] * lobe_px * ss) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        Lc = cv2.morphologyEx(Lb, cv2.MORPH_CLOSE, ker)
        Lc = cv2.morphologyEx(Lc, cv2.MORPH_OPEN, ker)
        Lb = np.where(hit, Lc, Lb).astype(F32)
    if o.get('smooth', 0.0):
        # simplify the light into big hand-shaped masses (painters merge small light / shadow islands)
        sg_ = o['smooth'] * lobe_px * ss
        Lb = np.where(hit, _blur(Lb * hitf, sg_) / (_blur(hitf, sg_) + 1e-4), Lb).astype(F32)
    if o['warp']:
        A_ = o['warp'] * lobe_px * ss * 0.25
        cells = max(W2 / (ss * lobe_px * 0.9), 3)
        wx = _noise(W2, H2, cells, seed + 51, 3) * A_
        wy = _noise(W2, H2, cells, seed + 52, 3) * A_
        gx1, gy1 = np.meshgrid(np.arange(W2, dtype=F32), np.arange(H2, dtype=F32))
        Lw = cv2.remap(Lb, gx1 + wx, gy1 + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Lb = np.where(hit, Lw, Lb)
    nz = _noise(W2, H2, max(W2 / (ss * lobe_px * 1.5), 3), seed + 11, 4) * o['noise']
    # fine, curly brush wobble on the plane borders (scalloped shadow edges, not ruler arcs)
    nz = nz + _noise(W2, H2, max(W2 / (ss * lobe_px * 0.35), 6), seed + 12, 2) * o['noise'] * o.get('curl', 0.6)
    x = Lb + nz
    st = o['step']
    s1 = _ss(o['t_lo'] - st, o['t_lo'] + st, x)
    s2 = _ss(o['t_hi'] - st, o['t_hi'] + st, x)
    upv = Nb @ np.asarray(up, F32)
    pal = dict(pal)
    for k_ in ('shade', 'sky', 'bounce', 'hi', 'lit_lo'):
        if o.get(k_) is not None:
            pal[k_] = _c(o[k_])
    if o.get('deep_col') is not None:
        pal['deep'] = _c(o['deep_col'])
    sh = pal['shade'][None, None, :] * np.ones_like(Nb)
    sh = sh + (pal['sky'] - pal['shade']) * (o['sky_fill'] * _ss(-0.2, 0.8, upv))[..., None]
    sh = sh + (pal['bounce'] - sh) * (o['bounce'] * _ss(0.0, 0.9, -upv))[..., None]
    core = _blur(_ss(0.8, 4.0, od * density) * hitf, lobe_px * ss * 1.5) * o['deep']
    sh = sh + (pal['deep'] - sh) * core[..., None]
    if o['shade_lift']:
        sh = sh + (pal['lit_lo'] - sh) * o['shade_lift']
    mid = pal['mid'] * (1 - o['mid_mix']) + pal['lit_lo'] * o['mid_mix']
    lit = _c(o['lit']) if o.get('lit') is not None else pal['lit']
    if o.get('mid') is not None:
        mid = _c(o['mid'])
    # inside each plane only a soft painted gradient: shade lightens toward its border, the lit plane
    # runs from a pale cool white at its border up to the warm cream / white of the crowns
    sh = sh * (1 + o['grad'] * 0.5 * _ss(o['t_lo'] - 0.25, o['t_lo'], x))[..., None]
    lit_in = (lit * 0.55 + mid * 0.45) if o.get('lit_in') is None else _c(o['lit_in'])
    kl = _ss(o['t_hi'] + st, o['t_hi'] + st + o.get('lit_span', 0.35), x)[..., None]
    litc = lit_in[None, None, :] * (1 - kl) + lit[None, None, :] * kl
    col = sh + (mid[None, None, :] - sh) * s1[..., None]
    col = col + (litc - col) * s2[..., None]
    if o['lit_warm']:
        # warm cream on the flank that faces the sun (screen direction warm_dir), cooler white elsewhere
        wd_ = np.asarray(o.get('warm_dir', (1.0, 0.0, -0.3)), F32)
        wd_ = wd_ / (np.linalg.norm(wd_) + 1e-6)
        wf = _ss(0.0, 0.6, Nb @ wd_) * s2
        col = col * (1 + o['lit_warm'] * wf[..., None] * np.array([0.02, -0.005, -0.08], F32))
    # sun direction on screen; highlights are smeared from their edge INTO the cloud (away from the sun)
    sd = np.array([Lf[0], Lf[1]], F32)
    sd = sd / (np.linalg.norm(sd) + 1e-6)
    gxs, gys = np.meshgrid(np.arange(W2, dtype=F32), np.arange(H2, dtype=F32))

    def smear(m, length, n=6, sgn=1.0):
        acc = m.copy()
        for i in range(1, n + 1):
            d = length * i / n * sgn
            sh_ = cv2.remap(m, gxs + sd[0] * d, gys + sd[1] * d, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
            acc = np.maximum(acc, sh_ * (1 - i / (n + 1)) ** 1.5)
        return acc * hitf

    # sunward silhouette limb: the painter's brightest white hugs the lit contour (gow_02)
    Dsil = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), 1e4).astype(F32)
    ga_ = _blur(hitf, max(lobe_px * ss * 0.15, 1.0))
    gy2, gx2 = np.gradient(ga_)
    gl2 = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-6
    sunward = _blur(np.clip((-(gx2 / gl2) * sd[0] - (gy2 / gl2) * sd[1]) * 1.4 + 0.1, 0, 1) * (Dsil < 3 * ss),
                    lobe_px * ss * 0.2)
    sunward = np.clip(sunward * 3.0, 0, 1)
    lpx = o.get('limb_px', 0.35) * lobe_px * ss
    limb = np.exp(-Dsil / max(lpx, 1.0)) * sunward
    hl = limb * o.get('limb', 0.7)
    if band is not None and edge and o.get('sep', 0.0):
        # the head BEHIND a crest outline sits in a soft painted shade band just beyond it (separates the
        # stacked heads, wwy_01) - broad, one-sided, never a pit
        bl0 = np.clip(band * 1.5, 0, 1)
        beyond = np.clip(smear(bl0, o.get('sep_px', 0.45) * lobe_px * ss, n=8, sgn=-1.0) - bl0, 0, 1)
        beyond = _blur(beyond, lobe_px * ss * 0.08)
        tgt_s = (mid * 0.5 + pal['shade'] * 0.5)[None, None, :]
        col = col + (tgt_s - col) * (beyond * o['sep'])[..., None]
    if band is not None and edge:
        bl = np.clip(band * 1.5, 0, 1)
        cl = smear(bl, o.get('crest_px', 0.3) * lobe_px * ss)
        hl = np.maximum(hl, cl * (o['crest_shade'] + (o['crest_lit'] - o['crest_shade']) * s2) * min(edge, 1.5))
    hl = np.clip(hl, 0, 1)
    lit_side = np.clip(s2 + 0.5 * s1, 0, 1)
    tgt = pal['hi'][None, None, :] * lit_side[..., None] + (pal['lit_lo'] * 0.7 + mid * 0.3)[None, None, :] * (1 - lit_side[..., None])
    col = col + (tgt - col) * hl[..., None]
    if o.get('florets', 0.0) and rl is not None:
        # painted cauliflower inside the masses: every medium floret gets a crisp stepped crescent - a
        # cool shade crescent on its side away from the sun and a bright cap toward it (never a smooth ball)
        fl0, fl1 = o.get('floret_range', (0.12, 0.9))
        wf = (_ss(fl0 * 0.7, fl0, rl / lobe_px) * (1 - _ss(fl1, fl1 * 1.4, rl / lobe_px)) * hitf).astype(F32)
        dn_ = Ns @ Lf
        nzf = _noise(W2, H2, max(W2 / (ss * lobe_px * 0.3), 6), seed + 17, 2) * 0.08
        sh_c = (1 - _ss(-0.05, 0.05, dn_ + o.get("floret_cut", 0.15) + nzf)) * wf
        hi_c = _ss(0.62, 0.72, dn_ + nzf) * wf
        sh_c = _blur(sh_c, 0.6 * ss)
        hi_c = _blur(hi_c, 0.6 * ss)
        amt = o['florets']
        lit_k = np.clip(s1 * 0.4 + s2 * 0.6, 0, 1)
        shc = (mid * (1 - o.get('floret_dark', 0.25)) + pal['shade'] * o.get('floret_dark', 0.25))[None, None, :]
        col = col + (shc - col) * (sh_c * amt * (0.35 + 0.65 * lit_k))[..., None]
        col = col + (np.clip(pal['hi'], 0, 1.1)[None, None, :] - col) * (hi_c * amt * 0.8 * lit_k)[..., None]
    if o.get('brush', 0.0):
        # brush texture: two crossed families of short strokes, value + a hint of hue shift (sky / lavender)
        cells = max(W2 / (ss * lobe_px * 0.5), 6)
        b1 = _noise(W2, H2, cells, seed + 71, 3, stretch=2.5, angle=o.get('brush_angle', 15))
        b2 = _noise(W2, H2, cells * 0.5, seed + 72, 2)
        bt = (0.7 * b1 + 0.5 * b2) * o['brush']
        tint = np.where(bt[..., None] > 0, (pal['sky'] * 0.5 + pal['shade'] * 0.5)[None, None, :],
                        np.asarray(o.get('brush_tint', (0.76, 0.72, 0.9)), F32)[None, None, :])
        col = col * (1 + 0.5 * bt[..., None]) + (tint - col) * (np.abs(bt) * 0.6)[..., None] * (1 - 0.6 * s2[..., None])
    vv = np.clip(s1 * 0.5 + s2 * 0.5, 0, 1)
    return col.astype(F32), vis, Nb, vv


def render_lobes(lobes, w, h, f, ppx, ppy, L, pal, ss=2, pn=(0.0, 1.0, 0.0), up=(0.0, -1.0, 0.0),
                 form=0.55, form_px=None, wrap=0.35, density=2.2, march=(20, 0.035, 1.12, 0.06),
                 bands=0.6, band_noise=0.06, terminator=0.35, rim=1.0, rim_back=1.0, glow=0.0,
                 fog=0.0, fog_ref=None, haze=0.0, lost=1.0, lost_px=None, base_soft=1.0, seed=0,
                 mottle=0.02, key=1.0, sky_fill=0.75, bounce=0.5, deep_amt=0.6, lit_curve=1.0, x0=0, y0=0,
                 lobe_px=None, inflate=0.75, inflate_px=None, head=0.6, head_px=None, micro=0.06, paint=1.0, dab_amt=0.025, warp=1.0, contrast=0.45, pivot=0.3, shadow_soft=0.35, edge=1.0, edge_px=None, rim_px=2.2, crest_focus=None, dimple=1.0, limb=0.25, limb_px=None, keep=None,
                 base_shade=0.0, crest_fade=None, rim_wrap=0.1, out=None, style='lobe', flat=None):
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
    if _DEBUG is not None:
        _DEBUG['hid'] = np.where(hit, hid, -1)
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
        if style == 'flat':
            band = _ss((flat or {}).get('jump0', 0.03), (flat or {}).get('jump1', 0.25), jump)
        if style == 'flat' and (flat or {}).get('head_lines', True):
            # crest lines only where one HEAD overlaps another (not around every floret on a head)
            hidm = np.where(hit, E[:, 11].astype(np.int64)[sidc], -1).astype(F32)
            hs = cv2.remap(hidm, gx0 + sdir[0] * off * 2.0, gy0 + sdir[1] * off * 2.0, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
            other = ((hs != hidm) & (hs >= 0)).astype(F32)
            other = cv2.dilate(other, np.ones((3, 3), np.uint8))
            band = band * other
        band = np.maximum(band, _blur(band, off * 0.35) * 0.9)
        band *= hitf * np.clip(Ns @ L.astype(F32) + 0.6, 0, 1)
        if crest_focus is not None:
            cx_, cw_, floor_ = crest_focus
            xs2 = (np.arange(W2, dtype=F32)[None, :] / ss + x0 - cx_) / cw_
            cn = _noise(W2, H2, max(W2 / (ss * lobe_px * 1.5), 3), seed + 61, 3)
            band = band * np.clip(floor_ + (1 - floor_) * np.exp(-xs2 * xs2) + 0.35 * cn, 0.05, 1.0)
        if crest_fade is not None:
            # hot crests are densest toward the sun / horizon and die out toward the foreground
            yf0, yf1, fl_ = crest_fade
            yy_ = (np.arange(H2, dtype=F32)[:, None] / ss + y0)
            band = band * (1 - (1 - fl_) * _ss(yf0, yf1, yy_))
        band_g = band
        if _DEBUG is not None:
            _DEBUG['band'] = band
    if style == 'flat':
        col, vis, N, vv = _paint_flat(hit, hitf, t0, od, _norm(Nb * form + Nh * head), Ns, L, pal, up, pn,
                                      band_g if edge else None, sidc, E, lobe_px, ss, seed, density, wrap,
                                      shadow_soft, base_shade, edge, flat, rl=rl)
    else:
        # flat cut bases seen from slightly below: a dark, flat underside (never a bright sliver)
        flatb = (hit & (Ns @ np.asarray(pn, F32) > 0.97)).astype(F32)
        if flatb.any():
            direct = direct * (1 - 0.85 * flatb)
        if base_shade:
            # value structure of a tall cloud: lit crown, darker body toward the base (screen rows of the hit)
            rows_ = np.nonzero(hit.any(1))[0]
            if len(rows_):
                yn_ = (np.arange(H2, dtype=F32) - rows_[0]) / max(rows_[-1] - rows_[0], 1)
                direct = direct * (1 - base_shade * _ss(0.3, 1.0, yn_))[:, None]
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
    # only silhouettes that face the sun (screen space) get the lining / backlit glow - no all-round halo
    ga_ = _blur(hitf, max(rpx * 1.5, 1.0))
    gy2, gx2 = np.gradient(ga_)
    gl2 = np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-6
    onx2, ony2 = -gx2 / gl2, -gy2 / gl2
    if L[2] > 0.05:
        sx_, sy_ = px2 + f2 * L[0] / L[2], py2 + f2 * L[1] / L[2]
        gxs, gys = np.meshgrid(np.arange(W2, dtype=F32), np.arange(H2, dtype=F32))
        ddx, ddy = sx_ - gxs, sy_ - gys
        dl_ = np.sqrt(ddx * ddx + ddy * ddy) + 1e-6
        ddx, ddy = ddx / dl_, ddy / dl_
    else:
        ddx, ddy = sdir[0], sdir[1]
    sunward = np.clip((onx2 * ddx + ony2 * ddy) * 1.3 + rim_wrap, 0, 1)
    rimv = thin * sunward * (rim * facing * np.clip(vis * 1.5, 0, 1) + rim_back * fwd * 2.5)
    if crest_fade is not None:
        yf0, yf1, fl_ = crest_fade
        rimv = rimv * (1 - (1 - fl_) * _ss(yf0, yf1, np.arange(H2, dtype=F32)[:, None] / ss + y0))
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
                cut_top=None, hang=0.0, levels=None, flat=1.0, core_flat=1.0, min_r=0.7,
                head_size=(0.08, 0.28, 1.8), head_mul=1.0, hier=0.0, front=0.8, sub_head=0.0, recede=0.0):
    """Build the lobe cluster of one cloud (camera space, units = plate px at the cloud's depth).

    X, Yb, Z: base centre (Yb = base height, y down); width / height; kind: 'cu' heap, 'cb' tower,
    'bank' low heap; lean: top offset / height; wobble: sideways meander of the column; depth: front
    to back thickness / width; detail: floret density multiplier; sun3: unit sun vector (florets
    favour the sun side); anvil: None or dict(left, right, thick, top) for a cumulonimbus anvil
    (px extents from the column axis; top = y of the flat top, default top of the column);
    heads: number of surface heads (default by size); hang: fraction of lobes allowed to hang below
    the base (breaks a ruler-straight base); head_size: (min, span, power) of head radii / half width;
    head_mul: head count multiplier; hier / front: see surface_grow. Returns a Lobes."""
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
    nh = heads if heads is not None else int(area * 12 * (0.5 + 0.5 * detail) * head_mul)
    for i in range(nh):
        s = rng.random() ** 0.8
        pw = float(_profile(kind, s)) * hw
        if pw < 0.02 * hw:
            continue
        phi = rng.uniform(-1.5, 1.5)
        rr = hw * (head_size[0] + head_size[1] * rng.random() ** head_size[2]) * (0.6 + 0.4 * pw / hw)
        k = pw - rr * rng.uniform(0.1, 0.5)
        ex = axis_x(s) + math.sin(phi) * k
        ez = Z - math.cos(phi) * k * depth + recede * s * hw
        ey = Yb - s * height
        if kind not in ('cb', 'tcu') or s > 0.8:
            ey = Yb - min(s * height, height - rr * 0.6)
        cores.append((np.array([ex, ey, ez]), np.array([rr * rng.uniform(1.0, 1.2), rr * core_flat, rr])))
    sizes = [hw * f_ for f_ in (levels if levels is not None else (0.12, 0.065, 0.034, 0.018))]
    sizes = [r_ for r_ in sizes if r_ >= min_r]
    area = math.pi * hw * height * 1.3
    lobes = surface_grow(rng, cores, sizes, dens=1.2 * detail, area=area, sun=sun3, flat=flat, clump=0.5,
                         front=front, hier=hier, sub_head=sub_head * hw)
    L.add_grown(lobes, cut_fn=lambda: dict(cut_lo=base_cut()), cut_hi=ct, scale=sc, haze=haze, warm=warm)
    if anvil is not None:
        anvil = dict(anvil)
        anvil.setdefault('col_hw', float(_profile(kind, 0.9)) * hw)
        _anvil(L, rng, axis_x(1.0), anvil.get('top', Yb - height), Z, anvil, sc, depth * hw, haze, warm, sun3)
    return L


def _anvil(L, rng, xc, top, Z, a, sc, zspread, haze, warm, sun3):
    """Flat-topped anvil (wwy_02) built in the same cauliflower language as the column: a slab of medium
    lobes filling a wedge profile (thickest over the column, a long thin downwind blade, a short blunt
    upwind end), cut flat on top, with a cone of smaller lobes flaring up from the column into its
    underside, then grown florets on the edges and the ragged underside."""
    left, right, th = a['left'], a['right'], a['thick']
    n = int(a.get('n', 90)) * 3
    down_right = right >= left
    zk = a.get('zk', 0.35)
    cores = []
    for i in range(n):
        u = (i + rng.random()) / n * 2 - 1                 # stratified along the anvil
        x = xc + (u * right if u > 0 else u * left)
        downwind = (u > 0) == down_right
        au = abs(u)
        t = th * (1 - 0.85 * au ** 1.2) if downwind else th * (1 - 0.45 * au ** 2)
        r = t * rng.uniform(0.35, 0.6)
        y = top + r * 0.55 + rng.uniform(0, max(t - r * 1.1, 0.0))
        z = Z + rng.uniform(-1, 1) * zspread * zk * (1 - 0.5 * au)
        cores.append((np.array([x, y, z]), np.array([r * 1.7, r, r * 1.3])))
    chw = a.get('col_hw', th * 2.5)
    for k in range(int(n * 0.35)):
        s_ = rng.random() ** 0.8                           # 0 = down on the column .. 1 = under the anvil
        side = -1 if rng.random() < 0.5 else 1
        ext = right if side > 0 else left
        x = xc + side * (chw * 0.55 + ext * 0.5 * s_ ** 2.0) * rng.uniform(0.55, 1.0)
        y = top + th * (0.7 + 3.0 * (1 - s_))
        r = th * (0.3 + 0.3 * s_) * rng.uniform(0.8, 1.2)
        cores.append((np.array([x, y, Z + rng.uniform(-1, 1) * zspread * zk * 0.6]), np.array([r * 1.3, r, r])))
    lob = surface_grow(rng, cores, [th * 0.28, th * 0.14, th * 0.07], dens=0.8, area=(left + right) * th * 4,
                       front=0.85, down_cut=0.7, flat=0.7, sun=sun3, clump=0.6, hier=0.8)
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
                       shape=None, anvil_thick=0.07, **kw):
    """Towering cumulus / cumulonimbus as an RGBA plate.

    cx, base_y: base centre (plate px); height: base to top (px); width: column width (default
    0.62 * height); anvil: flat-topped anvil at the top (anvil_dir +1 = spreads right); anvil_len:
    downwind extent / height; lean: top offset / height; haze: aerial haze 0..1; detail: floret
    density; eye_y: horizon (default base_y + 0.05 h: we look very slightly up at the base);
    persp: camera distance / height (smaller = stronger perspective); kind 'cb' (tower) | 'cu' heap.
    anvil_thick: anvil sheet thickness / height; shape: dict for cloud_lobes (hier, front, head_size,
    head_mul: scale hierarchy, see surface_grow). Extra keywords go to render_lobes (form, wrap,
    density, bands, rim, lost, base_shade, ...)."""
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
                  thick=anvil_thick * height, n=int(90 * detail))
    shp = dict(shape or {})
    lob = cloud_lobes(rng, X, Yb, dist, width, height * (0.9 if anvil else 1.0), kind=kind, lean=lean,
                      detail=detail, sun3=Ls, anvil=an, haze=haze, scale=0.5 * width, **shp)
    # secondary turrets: lower towers leaning against the main column -> stepped, grouped massing
    for k in range(int(turrets)):
        side = (-1) ** k * (1 if rng.random() < 0.5 else -1) if k else float(rng.choice([-1, 1]))
        th_ = height * rng.uniform(0.35, 0.7) * (0.85 ** k)
        tw_ = min(width * rng.uniform(0.55, 0.85), th_ * 0.85)
        off = side * (width * 0.5 + tw_ * rng.uniform(0.05, 0.3))
        dz = rng.uniform(-0.25, 0.35) * width
        sub = cloud_lobes(np.random.default_rng(seed * 31 + k + 1), X + off, Yb + rng.uniform(0, 0.02) * height,
                          dist + dz, tw_, th_, kind='tcu', lean=-side * rng.uniform(0.0, 0.12), detail=detail,
                          sun3=Ls, haze=haze, scale=0.5 * width, **shp)
        lob.extend(sub)
    pal = palette(preset)
    kw.setdefault('form_px', 0.09 * width)
    out = _render_side(lob, w, h, cam, Ls, pal, ss, seed=seed, **kw)
    if base_fade:
        dissolve_base(out, cx, width * 1.6, base_y, base_fade * height, seed=seed)
    return out


def cumulus_plate(w, h, clouds, sun=None, sun_dir=None, preset='noon', sun_z=0.15, detail=1.0, ss=1.5,
                  eye_y=None, persp=5.0, base_fade=0.12, shape=None, **kw):
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
                          sun3=Ls, haze=hz, depth=0.7, min_r=0.6 * k, scale=0.5 * max(cw, ch * 0.6) * k,
                          **dict(shape or {}))
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
                    stretch=2.2, opacity=0.85, sun_dir=(0.6, -0.8), patch=0.35, persp=1.8, rows=1.0,
                    avoid=(), soft=1.0, ripple=1.0, ss=2, body=None, lit=None):
    """Mackerel sky / altocumulus (gow_02, yn_02): small broken cloudlets organised in rippled rows.

    The field is laid out along a flow direction (`angle`, degrees; rows run along it) and the flecks
    are sheared along the wind. Size follows perspective: flecks at the top of `region` are `persp`
    times the size of those at its bottom (the sky plane recedes toward the horizon). Density is
    gathered into patches (`patch`, fraction of w) that fade out at their edges instead of stopping.
    Each fleck is a lumpy cluster of 2-4 ellipses with a crisp lit edge on the sun side (`sun_dir`,
    screen direction) and a soft, cooler trailing underside (`soft`).

    region: (y0, y1) band (fractions of h); density: coverage 0..1; size: mean fleck length at the
    bottom of the band (fraction of w); stretch: elongation; rows: row spacing multiplier; ripple:
    waviness of the rows; avoid: [(x, y, rx, ry), ...] ellipses (plate px) where the field thins out
    (keep the sky clean around a hero cloud); opacity: max alpha; body / lit: optional colour
    overrides. Deterministic per seed."""
    pal = palette(preset)
    rng = np.random.default_rng(seed)
    y0, y1 = region[0] * h, region[1] * h
    W2, H2 = int(w * ss), int(h * ss)
    m = np.zeros((H2, W2), np.float32)
    val = np.zeros((H2, W2), np.float32)
    big = _noise(w, h, 2.0 / max(patch, 1e-3), seed + 5, 3, stretch=4.0, angle=angle)
    lo_, hi_ = np.percentile(big, 15), np.percentile(big, 92)
    big = np.clip((big - lo_) / (hi_ - lo_ + 1e-6), 0, 1)
    a = math.radians(angle)
    ux, uy = math.cos(a), -math.sin(a)                 # along the rows (screen, y down)
    vx, vy = -uy, ux                                   # across the rows
    L0 = size * w
    cxs, cys = w / 2.0, (y0 + y1) / 2.0
    R = math.hypot(w, y1 - y0) * 0.6 + L0 * persp * 4
    ph = rng.uniform(0, 6.28, 4)
    SH = 4                                             # cv2 sub-pixel shift bits (x16) -> use 2 bits
    v = -R
    while v < R:
        yc = cys + v * vy
        k = 1.0 + (persp - 1.0) * float(np.clip((y1 - yc) / max(y1 - y0, 1), 0, 1))
        Ls = L0 * k
        u = -R + rng.uniform(0, Ls)
        while u < R:
            x = cxs + u * ux + v * vx
            y = cys + u * uy + v * vy
            y += ripple * Ls * (0.6 * math.sin(u / (Ls * 7.0) + ph[0] + v * 0.01) + 0.8 * rng.normal() / stretch)
            x += Ls * 0.3 * rng.normal()
            step = Ls * rng.uniform(0.8, 2.2)
            r1, r2 = rng.random(), rng.random()
            if 0 <= x < w and y0 - Ls < y < y1 + Ls:
                yy = int(min(max(y, 0), h - 1))
                xx = int(min(max(x, 0), w - 1))
                ky = (y - y0) / max(y1 - y0, 1)
                band = _ss(0.0, 0.15, ky) * (1 - _ss(0.8, 1.0, ky))
                pr = band * _ss(0.55 - 0.5 * density, 1.0, big[yy, xx])
                for (ax, ay, rx, ry) in avoid:
                    d2 = ((x - ax) / rx) ** 2 + ((y - ay) / ry) ** 2
                    pr *= float(np.clip(d2 - 0.6, 0, 1))
                if r1 < pr * 1.3:
                    frng = np.random.default_rng(int(r2 * 2 ** 31))
                    sc = Ls * frng.uniform(0.3, 1.0) ** 1.5 * 1.5 * (0.6 + 0.5 * pr)
                    op = frng.uniform(0.55, 1.0) * (0.5 + 0.5 * pr)
                    for q in range(int(frng.integers(3, 8))):
                        du = frng.uniform(-0.6, 0.6) * sc
                        dv = frng.uniform(-0.18, 0.18) * sc
                        ex = (x + du * ux + dv * vx) * ss
                        ey = (y + du * uy + dv * vy) * ss
                        ra = sc * frng.uniform(0.2, 0.45) * ss
                        rb = max(ra / stretch * frng.uniform(0.8, 1.3), 0.6 * ss)
                        ctr = (int(ex * 4), int(ey * 4))
                        ax_ = (int(ra * 4), int(rb * 4))
                        cv2.ellipse(m, ctr, ax_, -angle + frng.uniform(-35, 35), 0, 360, 1.0, -1, cv2.LINE_AA, 2)
                        cv2.ellipse(val, ctr, ax_, -angle, 0, 360, float(op), -1, cv2.LINE_AA, 2)
            u += step
        v += Ls * rows * rng.uniform(0.9, 1.5) / stretch * 2.2
    # break the ellipse outlines with a fine noise erosion (brushy, not stamped)
    n = _noise(W2, H2, max(W2 / (L0 * ss * 0.35), 8), seed + 3, 3, stretch=2.0, angle=angle)
    n2 = _noise(W2, H2, max(W2 / (L0 * ss * 0.12), 16), seed + 4, 2)
    m = np.clip((_blur(m, 0.8 * ss) - 0.4 + 0.4 * n + 0.25 * n2) * 3.0, 0, 1)
    d = np.asarray(sun_dir, np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    k = max(1.0, L0 * ss * 0.12)
    shp = _shift_img(m, -d[0] * k, -d[1] * k)
    rim = np.clip(m - shp, 0, 1)                             # sunward edge
    # soft cool trailing underside: the fleck smeared away from the sun
    tail = np.maximum(_shift_img(_blur(m, k * 1.2), -d[0] * k * 1.5, -d[1] * k * 1.5) * 0.55 * soft,
                      _blur(m, k * 0.6) * 0.8)
    m = _blur(m, 0.5 * ss)                                   # soft painted edge, not a stamped outline
    A = np.maximum(m * np.clip(0.75 + rim, 0, 1), tail * 0.45)
    A = A * np.clip(_blur(val, 1.0 * ss), 0, 1)
    A = np.maximum(A, _blur(m, k * 6) * 0.22)                # faint veil that joins a shoal into one patch
    bcol = _c(body) if body is not None else pal['sky'] * 0.6 + pal['lit_lo'] * 0.4
    lcol = _c(lit) if lit is not None else np.clip(pal['lit'], 0, 1.05)
    t = np.clip(rim * 0.6 + 1.3 * m, 0, 1)[..., None]
    col = bcol[None, None, :] * (1 - t) + lcol[None, None, :] * t
    col = _resize(col.astype(F32), w, h, cv2.INTER_AREA)
    A = _resize(A.astype(F32), w, h, cv2.INTER_AREA)
    a_ = np.clip(A * opacity, 0, 1)
    out = np.dstack([col, a_]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (a_ > 0.02).astype(F32), 3.0)
    return out


def stratus_plate(w, h, y, thick, preset='noon', seed=0, x_range=(-0.05, 1.05), count=6, length=(0.25, 0.7),
                  top_color=None, body_color=None, under_color=None, opacity=0.9, streak=1.0, sun_x=None):
    """Flat stratus / stratocumulus shelves along the horizon (wwy_01's layered horizon bars): long lens-
    shaped strata (length = fraction of w) whose tops are lit and slightly scalloped, with a flat, darker,
    softly lost underside and horizontal brush streaking inside.

    y: centre row of the layer (plate px); thick: max bar thickness (px); count: number of bars;
    top_color / body_color / under_color: override the palette (defaults: lit_lo, shade/sky mix, deep/
    shade mix); streak: horizontal brush texture; sun_x: bars are lit brighter toward this x."""
    pal = palette(preset)
    rng = np.random.default_rng(seed)
    top = _c(top_color) if top_color is not None else np.clip(pal['lit_lo'] * 0.6 + pal['lit'] * 0.4, 0, 1)
    body = _c(body_color) if body_color is not None else pal['shade'] * 0.5 + pal['sky'] * 0.5
    under = _c(under_color) if under_color is not None else pal['deep'] * 0.45 + pal['shade'] * 0.55
    xs = np.arange(w, dtype=F32)[None, :]
    ys = np.arange(h, dtype=F32)[:, None]
    A = np.zeros((h, w), F32)
    col = np.zeros((h, w, 3), F32) + body
    nz = _noise(w, h, max(w / (thick * 3.0), 4), seed + 1, 3)
    st = _noise(w, h, max(w / (thick * 1.2), 4), seed + 2, 3, stretch=25.0)
    for i in range(count):
        L = w * rng.uniform(*length)
        cx = w * rng.uniform(*x_range)
        cy = y + rng.uniform(-1.2, 1.2) * thick
        th = thick * rng.uniform(0.4, 1.0)
        u = (xs - cx) / (L / 2)
        prof = np.clip(1 - np.abs(u) ** 2.2, 0, 1) ** 0.6              # lens: thick middle, tapered ends
        tt = th * prof
        ytop = cy - tt * (0.75 + 0.25 * nz)                             # scalloped top
        ybot = cy + tt * 0.25                                           # flat base
        inside = _ss(ytop - 0.8, ytop + 0.8, ys) * (1 - _ss(ybot - th * 0.15, ybot + th * 0.35, ys))
        inside = inside * (tt > 0.3)
        vy = np.clip((ys - ytop) / np.maximum(ybot - ytop, 1e-3), 0, 1)  # 0 top .. 1 base
        c = top * (1 - _ss(0.0, 0.45, vy))[..., None] + body * (_ss(0.0, 0.45, vy) * (1 - _ss(0.5, 1.0, vy)))[..., None] \
            + under * _ss(0.5, 1.0, vy)[..., None]
        c = c * (1 + 0.06 * streak * st[..., None])
        if sun_x is not None:
            c = c * (1 + 0.08 * np.exp(-((xs - sun_x) / (0.3 * w)) ** 2))[..., None]
        a_i = np.clip(inside * opacity * (0.85 + 0.15 * st * streak), 0, 1)
        col = c * a_i[..., None] + col * (1 - a_i[..., None])
        A = a_i + A * (1 - a_i)
    out = np.dstack([col, A]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (A > 0.02).astype(F32), 4.0)
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


def sea_ground(w, h, horizon_y, x, dist, fov=55.0, cam_height=1.0):
    """Where a side-view tower standing on the sea of clouds should go: for screen column x and world
    distance `dist` (cam heights) returns (base_y, px_per_unit) - the screen row of the deck top there and
    the screen size of one cam-height at that distance (tower height px = height * px_per_unit)."""
    f, ppx, ppy, th = _pitch_cam(w, h, horizon_y, fov)
    X = (x - ppx) / f * dist * cam_height
    P = _to_cam(np.array([X, cam_height, dist * cam_height], np.float64), th)
    return float(ppy + f * P[1] / P[2]), float(f * cam_height / P[2])


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
    ldist = []
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
            ldist.append(dist)
    for c, r, Rd in floor:
        L.add(_to_cam(np.asarray(c, np.float64), th), r, scale=Rd * 3.0, dark=floor_dark)
        ldist.append(float(c[2]))
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
            ldist.append(dist)
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
    splits = [] if split is None else sorted([split] if np.isscalar(split) else list(split))
    E = L.array()
    ld = np.asarray(ldist, np.float64) / hc
    bidx = np.searchsorted(np.asarray(splits, np.float64), ld, side='right')   # 0 = nearest band
    near_plates = []

    def _subset(mask):
        idx = np.nonzero(mask)[0]
        remap = -np.ones(len(E), np.int64)
        remap[idx] = np.arange(len(idx))
        Es = E[idx].copy()
        rt = remap[Es[:, 11].astype(np.int64)]
        Es[:, 11] = np.where(rt >= 0, rt, np.arange(len(idx)))
        return Es, idx

    for b in range(len(splits)):
        nf = bidx == b
        if not nf.any():
            near_plates.append(np.zeros((h, w, 4), F32))
            continue
        # each band is painted from its own + all FARTHER lobes (nearer ones removed), so what the nearer
        # bands hide is painted too and parallax never opens holes; backlit shadows fall toward the camera,
        # so dropping the nearer lobes from the shadow march costs almost nothing
        Es, idx = _subset(bidx >= b)
        bbn = _bboxes(E[nf], f, ppx, ppy, w, h)
        yn0 = int(max(bbn[:, 1].min() - 0.12 * h, 0))
        pl = np.zeros((h, w, 4), F32)
        pl[yn0:] = render_lobes(Es, w, h - yn0, f, ppx, ppy, Ls, pal, keep=nf[idx], y0=yn0, **rk, **kw)
        pl[..., 3] = np.maximum(pl[..., 3], cv2.dilate(pl[..., 3], np.ones((3, 3), np.uint8)) * 0.95)
        pl[..., :3] = _bleed(pl[..., :3], (pl[..., 3] > 0.02).astype(F32), max(4.0, 0.01 * w))
        near_plates.append(pl)
    if splits:
        Es, idx = _subset(bidx == len(splits))
        res = render_lobes(Es, w, h, f, ppx, ppy, Ls, pal, **rk, **kw)
    else:
        res = render_lobes(E, w, h, f, ppx, ppy, Ls, pal, **rk, **kw)
    # painted far band (beyond the lobes) + fill: opaque from the horizon down
    ys = np.arange(h, dtype=F32)[:, None]
    hz_col = _c(haze_color) if haze_color is not None else pal['haze']
    deep = pal['deep'] * 0.6 + pal['shade'] * 0.4
    y_far = ppy + f * math.tan(math.atan(hc / (far_dist * hc)) - th)
    t = np.clip((ys - horizon_y) / max(y_far - horizon_y, 1), 0, 1)
    band = np.broadcast_to(hz_col * (1 - 0.35 * t[..., None]) + deep * 0.35 * t[..., None], (h, w, 3)).copy()
    # below the painted far band the fill (only ever seen through seams / gaps) is deep shadow, not haze
    ybeyond = _ss(y_far, y_far + 0.1 * h, ys)[..., None]
    band = band * (1 - ybeyond) + deep * ybeyond
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
    if splits:
        return [out] + near_plates[::-1]
    return out


# ----------------------------------------------------------------------------------------- sky streets


def sky_streets_plate(w, h, horizon_y, preset='noon', seed=0, region=(0.0, 0.5), fov=60.0, vp_x=None, size=0.02,
                      density=0.5, patch=1.0, streets=1.0, opacity=0.9, sun_dir=(0.6, -0.8), warm=0.25, avoid=(),
                      lit=None, body=None, warm_col=None, veil=0.35, ss=2, min_px=0.8, lumps=(2, 5), aspect=0.4,
                      dab_angle=-20.0, spacing=1.0, band_stretch=3.0, wisps=None, soft=1.0):
    """Altocumulus / mackerel sky laid out on a real sky plane in perspective (gow_02, yn_02).

    Flecks live on a horizontal plane above the camera, arranged in cloud STREETS (world lines that
    converge at a vanishing point on the horizon, vp_x) and gathered in world-space patches stretched
    along the streets, so size, spacing and foreshortening all fall off toward the horizon (larger
    overhead) and the flecks cluster into shoals with bare sky between. Each fleck is a painted dab
    (1-4 elongated, sheared ellipses, ragged edge, own opacity) - white on the sun side, a touch of sky
    blue on the underside, fading into the sky by ALPHA (no outline). A soft airbrushed veil glows
    under each shoal.

    horizon_y: horizon row (plate px); region: (top, bottom) band (fractions of h); fov: horizontal
    field of view; vp_x: vanishing point x (plate px, default centre); size: fleck length at the TOP of
    the region (fraction of w); density 0..1; patch: shoal scale multiplier; streets / spacing: street
    and along-street spacing multipliers; band_stretch: shoal elongation along the streets; warm:
    fraction of flecks with a pale warm tint; avoid: [(x, y, rx, ry)] ellipses where the field thins
    out; veil: shoal veil strength; dab_angle: mean screen tilt of the dabs (deg); lit / body /
    warm_col: colour overrides; min_px: flecks smaller than this are dropped. Straight-alpha RGBA."""
    pal = palette(preset)
    rng = np.random.default_rng(seed)
    if wisps is not None:
        veil = wisps
    f = (w / 2.0) / math.tan(math.radians(fov) / 2)
    cx = w / 2.0
    vpx = cx if vp_x is None else vp_x
    ang = math.atan((vpx - cx) / f)
    du = np.array([math.sin(ang), math.cos(ang)])     # along the street (world X, Z)
    dv = np.array([du[1], -du[0]])                    # across
    y0, y1 = region[0] * h, min(region[1] * h, horizon_y - 2)
    Zmin = f / max(horizon_y - y0, 1.0)
    Zmax = f / max(horizon_y - y1, 1.0)
    s_w = size * w * Zmin / f                         # world fleck size
    W2, H2 = int(w * ss), int(h * ss)
    m = np.zeros((H2, W2), F32)                       # opacity-weighted coverage
    tv = np.zeros((H2, W2), F32)                      # warm tint
    # shoals: smooth world-space field, stretched ALONG the streets
    K1 = rng.normal(size=(7, 2)) / (s_w * 10.0 * patch)
    K1[:, 0] /= band_stretch                          # component along du varies slowly
    P1 = rng.uniform(0, 6.28, 7)
    K2 = rng.normal(size=(5, 2)) / (s_w * 3.5 * patch)
    P2 = rng.uniform(0, 6.28, 5)

    def field(u, v):
        a = np.sin(K1[:, 0] * u + K1[:, 1] * v + P1).sum() / 7 * 2.2
        b = np.sin(K2[:, 0] * u + K2[:, 1] * v + P2).sum() / 5 * 2.2
        return 0.75 * a + 0.25 * b

    Xmax = (w * 0.65) / f * Zmax
    corners = np.array([[-Xmax, Zmin], [Xmax, Zmin], [-Xmax, Zmax], [Xmax, Zmax]])
    vs, us = corners @ dv, corners @ du
    sp_v = s_w * 1.5 * streets
    v = vs.min() + rng.uniform(0, sp_v)
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    S4 = ss * 4
    while v < vs.max():
        sv = rng.uniform(0.5, 1.0)
        u = us.min() + rng.uniform(0, s_w)
        wig = rng.uniform(0, 6.28)
        while u < us.max():
            vv_ = v + 0.3 * s_w * math.sin(u / (s_w * 5) + wig)
            P = du * u + dv * (vv_ + rng.normal() * 0.3 * s_w)
            X, Z = P
            su = s_w * rng.uniform(0.6, 1.5) * spacing
            r_a, r_b = rng.random(), rng.random()
            if Zmin * 0.95 < Z < Zmax * 1.05:
                x = cx + f * X / Z
                y = horizon_y - f / Z
                if -0.05 * w < x < 1.05 * w and y0 - 0.05 * h < y < y1 + 0.02 * h:
                    d = field(u, vv_)
                    ky = (y - y0) / max(y1 - y0, 1)
                    band = float(_ss(-0.05, 0.12, ky) * (1 - _ss(0.82, 1.0, ky)))
                    pr = float(_ss(0.9 - 1.1 * density, 1.2 - 0.6 * density, d)) * band * sv
                    for (ax, ay, rx, ry) in avoid:
                        d2 = ((x - ax) / rx) ** 2 + ((y - ay) / ry) ** 2
                        pr *= float(np.clip(d2 - 0.5, 0, 1))
                    if r_a < pr:
                        frng = np.random.default_rng(int(r_b * 2 ** 31))
                        rpx = s_w * f / Z * frng.uniform(0.35, 1.0) ** 1.2 * (0.6 + 1.0 * pr)
                        if rpx >= min_px:
                            fore = float(np.clip((horizon_y - y) / (horizon_y - y0 + 1e-6), 0.15, 1.0))
                            op = frng.uniform(0.55, 1.0) * (0.65 + 0.35 * pr)
                            tint = frng.random() < warm
                            for q in range(int(frng.integers(lumps[0], lumps[1] + 1))):
                                ox = frng.uniform(-0.6, 0.6) * rpx
                                oy = frng.uniform(-0.2, 0.2) * rpx * fore
                                ra = rpx * frng.uniform(0.35, 0.65)
                                rb = max(ra * aspect * (0.4 + 0.6 * fore) * frng.uniform(0.6, 1.2), 0.5)
                                c_ = (int((x + ox) * S4), int((y + oy) * S4))
                                ax_ = (max(int(ra * S4), 1), max(int(rb * S4), 1))
                                an_ = dab_angle + frng.uniform(-25, 25)
                                cv2.ellipse(m, c_, ax_, an_, 0, 360, float(op), -1, cv2.LINE_AA, 2)
                                if tint:
                                    cv2.ellipse(tv, c_, ax_, an_, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
            u += su
        v += sp_v * rng.uniform(0.7, 1.3)
    # ragged dab edges: erode the coverage with fine noise (brush texture), soft by alpha
    k = max(1.0, size * w * ss * 0.06)
    n1 = _noise(W2, H2, max(W2 / (size * w * ss * 0.35), 8), seed + 3, 3)
    cov = (m > 0).astype(F32)
    n2 = _noise(W2, H2, max(W2 / (size * w * ss * 0.12), 16), seed + 4, 2, stretch=3.0, angle=-dab_angle)
    cov = np.clip((_blur(cov, 0.5 * ss) - 0.35 + 0.45 * n1 + 0.25 * n2) * 2.5, 0, 1)
    # dry-brush streaks inside the dabs
    dry = np.clip(0.75 + 0.5 * _noise(W2, H2, max(W2 / (size * w * ss * 0.08), 16), seed + 6, 2, stretch=5.0,
                                       angle=-dab_angle), 0.3, 1.0)
    cov = cov * dry
    op = _blur(m, 0.5 * ss) / (_blur((m > 0).astype(F32), 0.5 * ss) + 1e-4)
    A = cov * np.clip(op, 0, 1)
    shp = _shift_img(cov, -sd[0] * k, -sd[1] * k)
    rim = np.clip(cov - shp, 0, 1)                    # sunward edge of each dab
    und = np.clip(cov - _shift_img(cov, sd[0] * k, sd[1] * k), 0, 1)   # underside
    vl = _blur(A, k * 8) * veil
    A = np.clip(np.maximum(A, vl * 0.6), 0, 1)
    lcol = _c(lit) if lit is not None else np.clip(pal['lit'], 0, 1.05)
    bcol = _c(body) if body is not None else pal['sky'] * 0.5 + pal['lit_lo'] * 0.5
    wcol = _c(warm_col) if warm_col is not None else np.clip(pal['lit'] * np.array([1.0, 0.95, 0.85], F32), 0, 1.05)
    tb = np.clip(und * 0.45 * soft + (1 - cov) * 0.5, 0, 1)[..., None]
    col = lcol[None, None, :] * (1 - tb) + bcol[None, None, :] * tb
    col = col + rim[..., None] * 0.08
    tw = np.clip(_blur(tv, 0.6 * ss), 0, 1)[..., None] * 0.7
    col = col * (1 - tw) + wcol[None, None, :] * tw
    col = _resize(col.astype(F32), w, h, cv2.INTER_AREA)
    A = _resize(A.astype(F32), w, h, cv2.INTER_AREA)
    a_ = np.clip(A * opacity, 0, 1)
    out = np.dstack([col, a_]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (a_ > 0.02).astype(F32), 3.0)
    return out


# ----------------------------------------------------------------------------------------- sea, lines, flare


def sea_plate(w, h, horizon_y, sun_x, near='#1d4f9a', far='#79b3dc', glitter=1.0, path_w=0.05, seed=0,
              haze=('#dbeefa', 0.5)):
    """Open sea below the horizon: deep saturated blue in the foreground rising to a pale far band,
    horizontal ripple strokes, a pale haze sheet on the horizon and a specular glitter path under the sun.
    Returns (plate RGBA, glitter) where glitter = (intensity map, phase map) for sea_glitter()."""
    rng = np.random.default_rng(seed)
    ys = np.arange(h, dtype=F32)[:, None]
    xs = np.arange(w, dtype=F32)[None, :]
    depth = np.clip((ys - horizon_y) / max(h - horizon_y, 1), 0, 1)          # 0 horizon .. 1 bottom
    t = depth ** 0.55
    col = _c(far)[None, None, :] * (1 - t[..., None]) + _c(near)[None, None, :] * t[..., None]
    col = np.broadcast_to(col, (h, w, 3)).copy()
    rip = _noise(w, h, max(w / 30.0, 4), seed + 1, 3, stretch=14.0)
    col = col * (1 + 0.05 * rip[..., None] * (0.3 + depth[..., None]))
    hc, ha = haze
    hz = np.exp(-(np.maximum(ys - horizon_y, 0) / (0.012 * h + 1))) * ha
    col = col + (_c(hc) - col) * hz[..., None]
    wpath = path_w * w * (0.3 + 1.7 * depth)
    kx = np.exp(-((xs - sun_x) / wpath) ** 2)
    n1 = _noise(w, h, max(w / 3.0, 8), seed + 2, 2, stretch=6.0)
    n2 = _noise(w, h, max(w / 7.0, 8), seed + 3, 2, stretch=3.0)
    dots = np.clip((n1 * 0.6 + n2 * 0.4 - 0.45) * 6, 0, 1) * (kx * 0.9 + 0.1 * np.exp(-((xs - sun_x) / (0.4 * w)) ** 2))
    dots = dots * (ys > horizon_y + 1) * (0.5 + 0.5 * (1 - depth))
    phase = rng.uniform(0, 6.28, (h // 4 + 1, w // 4 + 1)).astype(F32)
    phase = cv2.resize(phase, (w, h), interpolation=cv2.INTER_NEAREST)
    sheen = kx * (1 - depth) ** 1.5 * 0.18
    col = col + sheen[..., None] * np.array([0.9, 0.95, 1.0], F32)
    a = np.clip(ys - horizon_y + 0.5, 0, 1) * np.ones((1, w), F32)
    return np.dstack([col, a]).astype(F32), [dots.astype(F32) * glitter, phase]


def sea_glitter(glit, t, W, H, cam=(0.0, 0.0), zoom=1.0, depth=1.0, color=(1.0, 0.97, 0.88), rate=2.2):
    """Additive twinkling glitter for a sea_plate glitter list, sampled like drift() (no wind speed). Each
    dot has its own slow phase (coherent twinkle, no per-frame noise). Only the rows holding dots are
    updated (the glitter list caches its buffer)."""
    dots, phase = glit[0], glit[1]
    if len(glit) < 3:
        rows = np.nonzero(dots.max(1) > 0)[0]
        glit.append(int(rows[0]) if len(rows) else 0)
        glit.append(np.zeros(dots.shape, F32))
    y0, buf = glit[2], glit[3]
    tw = np.clip(0.5 + 0.5 * np.sin(rate * t + phase[y0:]), 0, 1) ** 3
    buf[y0:] = dots[y0:] * tw
    h, w = buf.shape
    z = 1 + (zoom - 1) * depth
    cx, cy = w / 2 + cam[0] * depth, h / 2 + cam[1] * depth
    M = np.array([[z, 0, W / 2 - cx * z], [0, z, H / 2 - cy * z]], np.float32)
    s_ = cv2.warpAffine(buf, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return s_[..., None] * (np.asarray(color, F32) * 1.4)


def power_lines_plate(w, h, pole_x, pole_top, seed=0, color=(0.1, 0.14, 0.24), wires=4, sag=0.06, ss=2,
                      left_y=None, second=True):
    """Foreground utility pole + sagging wires in silhouette (cm5_04): a tapered pole at pole_x rising
    from below the frame to pole_top, two crossarms with insulators, `wires` catenaries leaving to the
    left (left_y = their height there, default pole_top + 0.32 h). second: a smaller, farther pole on
    the left carrying the same wires. Returns straight-alpha RGBA."""
    W2, H2 = int(w * ss), int(h * ss)
    m = np.zeros((H2, W2), F32)
    S4 = 4 * ss

    def P(x, y):
        return (int(x * S4), int(y * S4))

    def pole(x, top, bot, wd, arm, sc):
        pts = np.array([[x - wd * 0.5, bot], [x + wd * 0.5, bot], [x + wd * 0.33, top], [x - wd * 0.33, top]])
        cv2.fillPoly(m, [np.round(pts * S4).astype(np.int32)], 1.0, cv2.LINE_AA, 2)
        ends = []
        for k, (ay, aw) in enumerate(((top + 0.02 * sc, arm), (top + 0.075 * sc, arm * 0.8))):
            th = max(0.009 * sc, 1.0)
            cv2.rectangle(m, P(x - aw, ay), P(x + aw, ay + th), 1.0, -1, cv2.LINE_AA, 2)
            for e in (-1, -0.45, 0.45, 1):
                ix = x + e * aw * 0.92
                cv2.rectangle(m, P(ix - th * 0.45, ay - th * 1.6), P(ix + th * 0.45, ay), 1.0, -1, cv2.LINE_AA, 2)
                ends.append((ix, ay - th * 1.6))
        cy = top + 0.16 * sc
        cv2.ellipse(m, P(x + wd * 0.9, cy), (max(int(wd * 0.75 * S4), 1), max(int(0.03 * sc * S4), 1)), 0, 0, 360,
                    1.0, -1, cv2.LINE_AA, 2)
        return ends

    sc = h * 0.9
    ends = pole(pole_x, pole_top, h * 1.1, 0.018 * w, 0.075 * w, sc)
    ly = left_y if left_y is not None else pole_top + 0.32 * h
    ends2 = None
    if second:
        x2 = 0.1 * w
        ends2 = pole(x2, ly - 0.07 * h, h * 1.1, 0.007 * w, 0.03 * w, sc * 0.4)
    ends = sorted(ends, key=lambda e: (round(e[1]), e[0]))
    ends2 = sorted(ends2, key=lambda e: (round(e[1]), e[0])) if ends2 else None
    pick = np.linspace(0, len(ends) - 1, wires).round().astype(int)
    th_ = max(int(round(0.0011 * w * ss)), 1)
    u = np.linspace(0, 1, 90)
    for i, j in enumerate(pick):
        ex, ey = ends[j]
        if ends2 is not None:
            tx, ty = ends2[j]
        else:
            tx, ty = -0.05 * w, ly + i * 0.01 * h
        xs_ = ex + (tx - ex) * u
        ys_ = ey + (ty - ey) * u + sag * h * (1 + 0.15 * i) * 4 * u * (1 - u)
        cv2.polylines(m, [np.round(np.stack([xs_, ys_], 1) * S4).astype(np.int32)], False, 1.0, th_, cv2.LINE_AA, 2)
        if ends2 is not None:
            xs3 = tx + (-0.08 * w - tx) * u
            ys3 = ty + 0.02 * h * u + sag * 0.3 * h * 4 * u * (1 - u)
            cv2.polylines(m, [np.round(np.stack([xs3, ys3], 1) * S4).astype(np.int32)], False, 1.0,
                          max(th_ - 1, 1), cv2.LINE_AA, 2)
    A = _resize(np.clip(m, 0, 1), w, h, cv2.INTER_AREA)
    col = np.broadcast_to(np.asarray(color, F32), (h, w, 3)).copy()
    return np.dstack([col, A]).astype(F32)


def shinkai_flare(W, H, lx, ly, intensity=1.0, tint=(1.0, 0.93, 0.8), t=0.0, star=1.0, ring=1.0, chain=1.0,
                  bloom=1.0, center=None, star_len=0.09, seed=5, vertical=0.0):
    """Shinkai sun flare (yn_02, wwy_05): the anime_flare glow + a hard six-point star, a faint
    prismatic ring, and a chain of small ghost dots / hexagons along the axis from the sun through the
    optical centre. vertical: optional thin vertical flare line with bead dots (yn_02). Additive (H, W, 3)."""
    out = _F.anime_flare(W, H, lx, ly, intensity=1.0, tint=tint, rays=6, glow=0.7 * bloom, ghosts=0.6 * chain,
                         halo=0.0, streak=0.25, rot=0.015 * t, starburst=1.2, ray_len=0.07, center=center)
    out = out + _F.glints(W, H, [lx], [ly], size=star_len, intensity=1.4 * star, color=tint, angle=0.26 + 0.01 * t,
                          arms=6)
    s = 2
    w, h = W // s, H // s
    xs, ys = np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))
    x0, y0 = lx / s, ly / s
    d = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / w
    add = np.zeros((h, w, 3), F32)
    if ring:
        R = 0.13
        rr = (d - R) / 0.012
        hue = np.stack([np.clip(0.5 + rr * 0.35, 0, 1), np.clip(1 - np.abs(rr) * 0.3, 0, 1),
                        np.clip(0.5 - rr * 0.35, 0, 1)], -1)
        add += np.exp(-rr * rr)[..., None] * hue * 0.045 * ring
    if chain:
        rng = np.random.default_rng(seed)
        cx_, cy_ = (w / 2, h / 2) if center is None else (center[0] / s, center[1] / s)
        vx, vy = cx_ - x0, cy_ - y0
        ks = np.sort(rng.uniform(0.15, 2.1, 11))
        cols = [np.array([1.0, 0.8, 0.55], F32), np.array([0.6, 0.9, 1.0], F32), np.array([0.8, 1.0, 0.7], F32),
                np.array([1.0, 0.7, 0.9], F32)]
        for i, k in enumerate(ks):
            gx, gy = x0 + vx * k, y0 + vy * k
            R = w * (0.004 + 0.018 * rng.random() ** 2.5)
            amp = 0.05 * rng.uniform(0.6, 1.3) * (1.0 if R < 0.012 * w else 0.5)
            b0x, b1x = int(max(gx - R * 2, 0)), int(min(gx + R * 2 + 1, w))
            b0y, b1y = int(max(gy - R * 2, 0)), int(min(gy + R * 2 + 1, h))
            if b1x <= b0x or b1y <= b0y:
                continue
            ex, ey = xs[b0y:b1y, b0x:b1x] - gx, ys[b0y:b1y, b0x:b1x] - gy
            rad = np.sqrt(ex * ex + ey * ey)
            if i % 3 == 1:
                th = np.arctan2(ey, ex) + 0.3
                seg = 2 * math.pi / 6
                rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            q = rad / R
            disc = _ss(1.05, 0.7, q) * (0.55 + 0.45 * _ss(0.3, 1.0, q))
            add[b0y:b1y, b0x:b1x] += disc[..., None] * cols[i % 4] * amp * chain
    if vertical:
        vx_ = np.exp(-((xs - x0) / 1.2) ** 2) * np.exp(-np.abs(ys - y0) / (0.35 * h))
        beads = (0.5 + 0.5 * np.cos((ys - y0) / (0.018 * h) * 2 * math.pi)) ** 8
        add += (vx_ * (0.4 + 0.6 * beads))[..., None] * np.asarray(tint, F32) * 0.35 * vertical
    add = cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
    return ((out + add) * intensity).astype(F32)


class SunFlare:
    """Cached shinkai_flare for animation: the sun-centred parts (glow, six-point star, prismatic ring) are
    rendered ONCE into a 2x canvas and translated to the sun each frame; the ghost chain (which slides
    along the axis through the optical centre) is re-painted per frame at half resolution, only inside
    each ghost's box. render() costs ~0.05-0.1 s at 1080p."""

    def __init__(self, W, H, tint=(1.0, 0.93, 0.8), star=1.0, ring=1.0, chain=1.0, bloom=1.0, star_len=0.09,
                 seed=5, vertical=0.0):
        self.W, self.H = W, H
        self.tint = np.asarray(tint, F32)
        self.chain, self.seed = chain, seed
        cw, ch = 2 * W, 2 * H
        # the sprite is rendered with the frame-size conventions (sizes are fractions of W)
        spr = _F.anime_flare(cw, ch, cw / 2, ch / 2, intensity=1.0, tint=tint, rays=6, glow=0.0,
                             ghosts=0.0, halo=0.0, streak=0.0, starburst=1.2, ray_len=0.035, center=None)
        spr = spr + _F.glints(cw, ch, [cw / 2], [ch / 2], size=star_len * 0.5, intensity=1.4 * star, color=tint,
                              angle=0.26, arms=6)
        yy, xx = np.mgrid[0:ch, 0:cw].astype(F32)
        d = np.sqrt((xx - cw / 2) ** 2 + (yy - ch / 2) ** 2) / W
        # glow (same profile as fx.anime_flare, normalised by the FRAME width) + a thin anamorphic streak
        core = 1.0 / (1.0 + (d / 0.004) ** 2.4)
        broad = 1.0 / (1.0 + (d / 0.022) ** 2) * 0.1 + np.exp(-d / 0.12) * 0.06 + np.exp(-d / 0.35) * 0.03
        spr = spr + (core[..., None] * np.array([1.0, 0.98, 0.94], F32) * 0.9 + broad[..., None] * self.tint) * 0.7 * bloom
        dx_, dy_ = np.abs(xx - cw / 2) / W, np.abs(yy - ch / 2) / H
        stk = np.exp(-(dy_ / 0.004) ** 2) * (np.exp(-dx_ / 0.22) * 0.6 + np.exp(-dx_ / 0.05) * 0.5)
        spr = spr + stk[..., None] * np.array([0.8, 0.9, 1.0], F32) * 0.3 * 0.25
        if ring:
            rr = (d - 0.13) / 0.012
            hue = np.stack([np.clip(0.5 + rr * 0.35, 0, 1), np.clip(1 - np.abs(rr) * 0.3, 0, 1),
                            np.clip(0.5 - rr * 0.35, 0, 1)], -1)
            spr = spr + np.exp(-rr * rr)[..., None] * hue * 0.045 * ring
        if vertical:
            vx_ = np.exp(-((xx - cw / 2) / 1.5) ** 2) * np.exp(-np.abs(yy - ch / 2) / (0.14 * H))
            beads = (0.5 + 0.5 * np.cos((yy - ch / 2) / (0.018 * H) * 2 * math.pi)) ** 8
            spr = spr + (vx_ * (0.4 + 0.6 * beads))[..., None] * self.tint * 0.35 * vertical
        self.spr = spr.astype(F32)
        rng = np.random.default_rng(seed)
        self.ghosts = [(float(k), W * (0.004 + 0.018 * rng.random() ** 2.5), 0.05 * rng.uniform(0.6, 1.3))
                       for k in np.sort(rng.uniform(0.15, 2.1, 11))]
        self.cols = [np.array(c, F32) for c in ((1.0, 0.8, 0.55), (0.6, 0.9, 1.0), (0.8, 1.0, 0.7), (1.0, 0.7, 0.9))]

    def render(self, lx, ly, intensity=1.0, t=0.0, center=None):
        W, H = self.W, self.H
        rot = 0.015 * t
        c, s_ = math.cos(rot), math.sin(rot)
        cw, ch = 2 * W, 2 * H
        # dst = R (src - c_src) + (lx, ly)
        M = np.array([[c, -s_, lx - (c * cw / 2 - s_ * ch / 2)], [s_, c, ly - (s_ * cw / 2 + c * ch / 2)]], np.float32)
        out = cv2.warpAffine(self.spr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.chain:
            cx_, cy_ = (W / 2, H / 2) if center is None else center
            vx, vy = cx_ - lx, cy_ - ly
            for i, (k, R, amp) in enumerate(self.ghosts):
                gx, gy = lx + vx * k, ly + vy * k
                b0x, b1x = int(max(gx - R * 2, 0)), int(min(gx + R * 2 + 1, W))
                b0y, b1y = int(max(gy - R * 2, 0)), int(min(gy + R * 2 + 1, H))
                if b1x <= b0x or b1y <= b0y:
                    continue
                ey, ex = np.mgrid[b0y:b1y, b0x:b1x].astype(F32)
                ex, ey = ex - gx, ey - gy
                rad = np.sqrt(ex * ex + ey * ey)
                if i % 3 == 1:
                    th = np.arctan2(ey, ex) + 0.3
                    seg = 2 * math.pi / 6
                    rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
                q = rad / R
                disc = _ss(1.05, 0.7, q) * (0.55 + 0.45 * _ss(0.3, 1.0, q))
                a_ = amp * (1.0 if R < 0.012 * W else 0.5) * self.chain
                out[b0y:b1y, b0x:b1x] += disc[..., None] * self.cols[i % 4] * a_
        return out * intensity


def deck_rows_plate(w, h, horizon_y, sun_x, preset='sunrise', seed=0, fov=55.0, cam_height=1.0, z_near=1.6,
                    z_far=60.0, row_step=0.16, head_w=0.9, head_h=0.35, rim_px=2.2, rim_focus=0.28, rim_floor=0.06,
                    top_col=None, body_col=None, deep_col=None, rim_col=None, far_col=None, gap=0.25,
                    gap_col=None, haze=0.55, split=None, ss=2, near_dark=0.55, glow=0.5, curl=1.0, top_band=0.35,
                    rim_near=1.0, brush=1.0, texture=0.06):
    """Sea of clouds as painted receding ROWS (yn_02 / wwy_02): painted far-to-near, every row is a band of
    domed heads (power-law widths, small cauliflower curls on the crests) whose scalloped top edge catches
    the low sun as a crisp rim; the bodies are flat violet planes that darken downward and toward the
    viewer, so each nearer row reads in front of the one behind it. Rows get smaller, flatter and denser
    toward the horizon, where they melt into a bright haze band; hot gold rims gather toward the sun
    column (rim_focus = width of the hot zone, fraction of w) and die out to cool, sparse edges in the
    foreground (rim_floor). Occasional dark-blue gaps open between rows (gap = fraction of rows).

    horizon_y: horizon row (plate px); sun_x: sun column (plate px); z_near / z_far: nearest / farthest row
    distance (cam heights); row_step: row spacing as a fraction of distance (log-spaced rows); head_w /
    head_h: head width / height in cam heights; rim_px: rim width at 1080p for the NEAREST rows (scales
    with row size, min ~1 px); split: list of distances -> returns [far, ..., nearest] plates (each band
    holds the rows in its range) for parallax; otherwise one plate. Straight-alpha RGBA (h, w, 4)."""
    pal = palette(preset)
    rng = np.random.default_rng(seed)
    f = (w / 2.0) / math.tan(math.radians(fov) / 2)
    top = _c(top_col) if top_col is not None else pal['mid'] * 0.6 + pal['lit_lo'] * 0.4
    body = _c(body_col) if body_col is not None else pal['shade']
    deep = _c(deep_col) if deep_col is not None else pal['deep']
    rimc = _c(rim_col) if rim_col is not None else pal['rim']
    farc = _c(far_col) if far_col is not None else pal['haze']
    gapc = _c(gap_col) if gap_col is not None else deep * 0.8
    sc = w / 1920.0
    # row distances, far -> near
    zs = []
    z = z_near
    while z < z_far:
        zs.append(z)
        z *= 1.0 + row_step * rng.uniform(0.7, 1.3)
    zs = zs[::-1]
    bounds = sorted(split) if split is not None else []
    nb = len(bounds) + 1
    W2, H2 = int(w * ss), int(h * ss)
    cols = [np.zeros((H2, W2, 3), F32) for _ in range(nb)]
    alps = [np.zeros((H2, W2), F32) for _ in range(nb)]
    xs = (np.arange(W2, dtype=F32) + 0.5) / ss
    ys_all = (np.arange(H2, dtype=F32) + 0.5) / ss
    hy = horizon_y
    for z in zs:
        bi = int(np.searchsorted(np.asarray(bounds, np.float64), z, side='right'))
        bi = nb - 1 - bi                       # 0 = far plate
        ppu = f / z                            # px per cam-height at this distance
        yb = hy + f * cam_height / z           # row base line (deck top at this distance)
        kz = float(np.clip((z - z_near) / (z_far - z_near), 0, 1)) ** 0.5     # 0 near .. 1 far
        # heads along the row
        hw_ = head_w * ppu
        hh_ = head_h * ppu * (0.55 + 0.45 * (1 - kz))          # far rows flatter
        crest = np.full(W2, -1e9, F32)
        x = -rng.uniform(0, hw_) - 0.1 * w
        while x < w * 1.1:
            wd = hw_ * rng.uniform(0.35, 1.0) ** 1.6 * 1.6
            ht = hh_ * rng.uniform(0.35, 1.0) * (wd / hw_) ** 0.35
            cx_ = x + wd * 0.5
            u = (xs - cx_) / (wd * 0.5)
            dome = np.sqrt(np.clip(1 - u * u, 0, 1)) ** 0.8 * ht
            # cauliflower curls on the crest (small domes)
            if curl and wd > 6 * sc:
                nc = int(rng.integers(2, 6))
                for _ in range(nc):
                    cw = wd * rng.uniform(0.08, 0.22)
                    cc = cx_ + rng.uniform(-0.45, 0.45) * wd
                    uu = (xs - cc) / (cw * 0.5)
                    base_h = np.sqrt(np.clip(1 - ((cc - cx_) / (wd * 0.5)) ** 2, 0, 1)) ** 0.8 * ht
                    dome = np.maximum(dome, np.where(np.abs(uu) < 1, base_h * 0.9 + np.sqrt(np.clip(1 - uu * uu, 0, 1)) * cw * 0.35, -1e9))
            crest = np.maximum(crest, np.where(np.abs(u) < 1, dome, -1e9))
            x += wd * rng.uniform(0.45, 0.8)
        ytop = yb - np.maximum(crest, 0.0) - hh_ * 0.15
        depth_px = max(hh_ * 2.5, 2.0 / ss)
        y0 = int(max(np.floor((min(ytop.min(), yb - hh_ * 2.6) - 2) * ss), 0))
        y1 = int(min(np.ceil((yb + depth_px) * ss), H2))
        if y1 <= y0:
            continue
        ys = ys_all[y0:y1, None]
        # coverage: crisp crest (anti-aliased), body fades out below (airbrushed) into the row in front
        aa = 0.7 / ss
        cov = np.clip((ys - ytop[None, :]) / aa + 0.5, 0, 1)
        fade = 1 - _ss(yb + depth_px * 0.2, yb + depth_px, ys)
        a = cov * fade
        # value: light at the crest, falling to the body / deep colour downward and toward the viewer
        # shade by depth below a SMOOTHED crest (no per-column streaks under the curls)
        ysm = cv2.GaussianBlur(ytop.reshape(1, -1).astype(F32), (0, 0), max(hw_ * 0.25 * ss, 1.0)).ravel()
        ysm = np.minimum(ysm, yb - hh_ * 0.1)
        dcrest = np.clip((ys - ysm[None, :]) / max(hh_ * 1.4, 1.0 / ss), 0, 1)
        nearness = 1 - kz
        bcol = body * (1 - near_dark * nearness * 0.5) + deep * near_dark * nearness * 0.5
        tk = _ss(0.0, top_band, dcrest)[..., None]
        foc0 = np.exp(-((xs - sun_x) / (rim_focus * w * 1.5)) ** 2)[None, :, None]
        topc = top[None, None, :] * (0.55 + 0.45 * max(kz, 0.0) ** 0.7 + 0.3 * foc0) + bcol[None, None, :] * 0.0
        topc = np.minimum(topc, top[None, None, :] * 1.15)
        c = topc * (1 - tk) + bcol[None, None, :] * tk
        c = c * (1 - 0.25 * nearness * _ss(0.5, 1.0, dcrest))[..., None] + deep[None, None, :] * (0.25 * nearness * _ss(0.5, 1.0, dcrest))[..., None]
        # rim: crisp lit crest line toward the sun, dying off sideways and toward the viewer
        rp = max(rim_px * sc * (ppu / (f / z_near)) ** 0.35, 0.6)
        foc = np.exp(-((xs - sun_x) / (rim_focus * w)) ** 2)
        side = np.clip(rim_floor + (1 - rim_floor) * foc * (rim_near + (1 - rim_near) * kz ** 0.5), 0, 1)
        side = side * (0.4 + 0.6 * np.clip(_noise(W2, 1, max(w / (hw_ + 1) * 0.8, 2), seed + int(z * 100), 2)[0] + 0.6, 0, 1))
        rim = np.exp(-np.maximum(ys - ytop[None, :], 0) / rp) * cov * side[None, :]
        c = c * (1 - np.clip(rim, 0, 1))[..., None] + rimc[None, None, :] * np.clip(rim, 0, 1)[..., None]
        # glow bleeding under the rim (backlit forward scatter)
        c = c + rimc[None, None, :] * (glow * 0.25 * side[None, :] * np.exp(-np.maximum(ys - ytop[None, :], 0) / (rp * 6)) * cov)[..., None]
        # head-to-head value variety (some heads catch more light) + horizontal brush texture
        if brush:
            hv = _noise(W2, 1, max(w / (hw_ + 1) * 1.2, 2), seed + 7 + int(z * 37), 2)[0]
            c = c * (1 + 0.12 * brush * hv[None, :, None])
        # aerial perspective toward the horizon
        hz = haze * kz ** 1.5
        c = c * (1 - hz) + farc[None, None, :] * hz
        cp, ap = cols[bi][y0:y1], alps[bi][y0:y1]
        # occasional gap: the deck opens just behind this row onto a darker, deeper blue layer
        if rng.random() < gap:
            gx = rng.uniform(0.05, 0.95) * w
            gwid = hw_ * rng.uniform(1.5, 4.0)
            gyc = yb - hh_ * 1.0
            gm = np.exp(-((xs - gx) / gwid) ** 2 - ((ys - gyc) / (hh_ * 0.7)) ** 2)
            gm = np.clip(gm * 1.6 - 0.3, 0, 0.9)
            gcol = gapc * (1 - hz) + farc * hz
            cp[:] = gcol[None, None, :] * gm[..., None] + cp * (1 - gm[..., None])
            ap[:] = gm + ap * (1 - gm)
        cp[:] = c * a[..., None] + cp * (1 - a[..., None])
        ap[:] = a + ap * (1 - a)
    out = []
    if texture:
        tx = _noise(W2, H2, max(w / 40.0, 4), seed + 91, 3, stretch=6.0, angle=-6) * texture
    for bi in range(nb):
        if texture:
            cols[bi] *= (1 + tx[..., None])
        A = _resize(alps[bi], w, h, cv2.INTER_AREA)
        RGB = _resize(cols[bi], w, h, cv2.INTER_AREA)
        RGB = RGB / np.maximum(A, 1e-4)[..., None] * (A > 1e-4)[..., None] + RGB * (A <= 1e-4)[..., None]
        RGB = _bleed(RGB.astype(F32), (A > 0.02).astype(F32), max(3.0, 0.004 * w))
        out.append(np.dstack([RGB, A]).astype(F32))
    return out if split is not None else out[0]


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


# =========================================================================================================
# PAINTED HEADS ENGINE (final round) - 2D painter's model of a Shinkai cumulus
# =========================================================================================================
# A cloud is an ordered list of HEADS (back -> front). A head is a core ellipse whose visible upper arc is
# covered with cauliflower BUMPS (two scales, power-law sizes, cusped notches between them). Heads are
# painted the way an artist stacks them: a lower / nearer head is painted later, so its crisp lit crown
# sits over the cool underside of the head behind it. Light has two layers:
#   * a BIG-FORM field computed once from the union silhouette (blurred-mask normal x screen-space sun
#     march, darker toward the base): the whole tower gets one lit flank and one shade flank;
#   * each head's OWN pseudo-3D normal (ellipsoid, blended with the winning bump's normal near the rim)
#     adds a lit crown and a shaded underside bound to that head. Small heads get less own contrast, so
#     they read as detail inside the big planes, never as pits.
# The value is stepped into a few painted planes (brush-warped borders), mapped through a warm/cool ramp,
# and silhouettes get an anisotropic edge: crisp on the sunward / upper side, lost (soft) on the down /
# shadow side and at the base.

_HCOLS = 27      # head record length
# head record: 0 cx 1 cy 2 rx 3 ry 4 rz 5 k(own contrast) 6 base_y 7 base_soft 8 top_y 9 haze 10 tone
#              11 aa_crisp 12 aa_soft 13 rim_amt 14 rim_px 15 warm 16 b0 17 nb 18 x0 19 y0 20 x1 21 y1
#              22 beta (bump normal weight) 23 glow 24-26 per-head light vector (0,0,0 = the global one)


@njit(cache=True, fastmath=True)
def _head_sdf(px, py, cx, cy, rx, ry, bumps, b0, nb, bthr):
    dx = (px - cx) / rx
    dy = (py - cy) / ry
    q = math.sqrt(dx * dx + dy * dy) + 1e-9
    g = math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2) / q + 1e-9
    sdf = (q - 1.0) / g
    for j in range(b0, b0 + nb):
        if bumps[j, 2] < bthr:
            continue
        ddx = px - bumps[j, 0]
        ddy = py - bumps[j, 1]
        s = math.sqrt(ddx * ddx + ddy * ddy) - bumps[j, 2]
        kk = min(0.3 * bumps[j, 2], 6.0)
        hh_ = max(kk - abs(sdf - s), 0.0) / (kk + 1e-9)
        sdf = min(sdf, s) - hh_ * hh_ * kk * 0.25
    return sdf


@njit(parallel=True, cache=True, fastmath=True)
def _paint_heads_k(P, A, Vbig, D, heads, bumps, Lx, Ly, Lz, wrap, alpha_only, lost_up, dscale, kin, crust, cmix, soft_in, Nz, namp, smk, rim_sil, rim_up, cbthr):
    """P: (H, W, 5) premultiplied accumulators (value, rim, haze, warm, glow); A: (H, W) alpha.
    Heads are composited in order with straight 'over'."""
    Hh, Ww = A.shape
    ls = math.sqrt(Lx * Lx + Ly * Ly) + 1e-9
    sx, sy = Lx / ls, Ly / ls
    for i in range(heads.shape[0]):
        cx, cy, rx, ry, rz = heads[i, 0], heads[i, 1], heads[i, 2], heads[i, 3], heads[i, 4]
        k = heads[i, 5]
        base_y, base_soft, top_y = heads[i, 6], heads[i, 7], heads[i, 8]
        haze, tone = heads[i, 9], heads[i, 10]
        aa0, aa1 = heads[i, 11], heads[i, 12]
        rim_amt, rim_px, warm = heads[i, 13], heads[i, 14], heads[i, 15]
        b0, nb = int(heads[i, 16]), int(heads[i, 17])
        x0 = max(int(heads[i, 18]), 0)
        y0 = max(int(heads[i, 19]), 0)
        x1 = min(int(heads[i, 20]), Ww)
        y1 = min(int(heads[i, 21]), Hh)
        beta = heads[i, 22]
        glow = heads[i, 23]
        if x1 <= x0 or y1 <= y0:
            continue
        hLx, hLy, hLz = Lx, Ly, Lz
        if heads[i, 24] != 0.0 or heads[i, 25] != 0.0 or heads[i, 26] != 0.0:
            hLx, hLy, hLz = heads[i, 24], heads[i, 25], heads[i, 26]
        hls = math.sqrt(hLx * hLx + hLy * hLy) + 1e-9
        hsx, hsy = hLx / hls, hLy / hls
        for y in prange(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                px = x + 0.5
                dx = (px - cx) / rx
                dy = (py - cy) / ry
                q = math.sqrt(dx * dx + dy * dy) + 1e-9
                g = math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2) / q + 1e-9
                sdf = (q - 1.0) / g
                ex, ey = dx / q, dy / q
                if q < 1.0:
                    wz = rz * math.sqrt(1.0 - q * q)
                    wnx, wny, wnz = dx, dy, math.sqrt(1.0 - q * q)
                else:
                    wz = -1e9
                    wnx, wny, wnz = ex, ey, 0.0
                for j in range(b0, b0 + nb):
                    bx, by, br, bz = bumps[j, 0], bumps[j, 1], bumps[j, 2], bumps[j, 3]
                    ddx = px - bx
                    ddy = py - by
                    d = math.sqrt(ddx * ddx + ddy * ddy) + 1e-9
                    s = d - br
                    if s < sdf:
                        ex, ey = ddx / d, ddy / d
                    # smooth union: fills the tiny pockets where two bumps and the core meet
                    kk = min(0.3 * br, smk)
                    hh_ = max(kk - abs(sdf - s), 0.0) / (kk + 1e-9)
                    sdf = min(sdf, s) - hh_ * hh_ * kk * 0.25
                    if d < br:
                        h_ = math.sqrt(br * br - d * d)
                        z = bz + h_
                        if z > wz:
                            wz = z
                            wnx, wny, wnz = ddx / br, ddy / br, h_ / br
                if namp > 0.0:
                    sdf += Nz[y, x] * namp
                isbase = False
                if base_y < 1e8:
                    sb = py - base_y
                    if sb > sdf:
                        sdf = sb
                        ex, ey = 0.0, 1.0
                        isbase = True
                if top_y > -1e8:
                    st = top_y - py
                    if st > sdf:
                        sdf = st
                        ex, ey = 0.0, -1.0
                facing = ex * hsx + ey * hsy
                lost = 0.75 * ey - 0.3 * facing + 0.05
                if lost_up > 0.0:
                    lost = max(lost, -ey * lost_up)
                lost = min(max(lost, 0.0), 1.0)
                aa1e = aa1
                if not alpha_only:
                    # inside the body a head's underside melts into the shade below it (no balloon outline)
                    ins0 = math.exp(-D[y, x] / dscale)
                    aa1e = aa1 + (soft_in * min(rx, ry) - aa1) * (1.0 - ins0) * max(ey, 0.0)
                aa = aa0 + (aa1e - aa0) * lost
                if isbase:
                    aa = max(base_soft, aa0)
                if sdf > 0.5 * aa0:
                    continue
                # crisp edges: symmetric AA; lost edges fade INWARD (the silhouette never bloats)
                a = min(max((0.5 * aa0 - sdf) / aa, 0.0), 1.0)
                if aa > aa0 * 1.5:
                    a = a * a * (3.0 - 2.0 * a)
                if a <= 0.0:
                    continue
                if alpha_only:
                    A[y, x] = a + A[y, x] * (1.0 - a)
                    continue
                # own normal: softened head ellipsoid blended with the winning primitive
                f_ = 1.1
                # detail lives on the contour: bump normals fade out with depth inside the union
                ins = math.exp(-D[y, x] / dscale)
                beta_e = beta * ins
                k_e = k * (kin + (1.0 - kin) * ins)
                # smooth, kink-free head normal (no clamp ring outside the core)
                hnx, hny, hnz = dx * f_, dy * f_, 1.0
                nx = hnx * (1 - beta_e) + wnx * beta_e
                ny = hny * (1 - beta_e) + wny * beta_e
                nz = hnz * (1 - beta_e) + wnz * beta_e
                nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
                lam = (nx * hLx + ny * hLy + nz * hLz) / nn
                lam = min(max((lam + wrap) / (1.0 + wrap), 0.0), 1.0)
                # lit crust: depth inside this head measured toward the sun (sphere-traced), so the
                # terminator follows the scalloped outline instead of a sphere's round shadow line
                own = lam
                if cmix > 0.0:
                    tt = 0.0
                    lim = crust * 3.0 * min(rx, ry)
                    bthr = max(0.14 * (1.0 - ins), cbthr) * min(rx, ry)
                    qx, qy = px, py
                    if sdf > -1.0:
                        qx, qy = px - ex * (sdf + 1.0), py - ey * (sdf + 1.0)
                    sd_ = min(_head_sdf(qx, qy, cx, cy, rx, ry, bumps, b0, nb, bthr), -0.01)
                    it = 0
                    while sd_ < 0.0 and tt < lim and it < 40:
                        stp = max(-sd_ * 0.7, 0.6)
                        tt += stp
                        sd_ = _head_sdf(qx + hsx * tt, qy + hsy * tt, cx, cy, rx, ry, bumps, b0, nb, bthr)
                        if base_y < 1e8:
                            sd_ = max(sd_, qy + hsy * tt - base_y)
                        it += 1
                    cr = math.exp(-tt / (crust * min(rx, ry) + 1e-6))
                    own = lam * (1.0 - cmix) + cr * cmix
                vb = Vbig[y, x]
                v = vb * (1.0 - k_e) + k_e * own * (0.3 + 0.7 * vb) + tone
                e = -sdf
                rim = 0.0
                gl = 0.0
                if e > -aa:
                    # head-level facing (smooth over the head) blended with the local primitive's facing:
                    # the lining runs continuously along a crest, the glow has no per-bump discs
                    hq = math.sqrt(dx * dx + dy * dy) + 1e-9
                    hfx, hfy = dx / hq, dy / hq
                    hfacing = hfx * hsx + hfy * hsy
                    fr = min(max((0.5 * facing + 0.5 * hfacing) * 1.6 - 0.25, 0.0), 1.0)
                    fr *= (1.0 - rim_up) + rim_up * min(max(-(0.5 * ey + 0.5 * hfy) * 1.4 + 0.1, 0.0), 1.0)
                    frg = min(max(hfacing * 1.4 - 0.2, 0.0), 1.0)
                    frg *= (1.0 - rim_up) + rim_up * min(max(-hfy * 1.4 + 0.1, 0.0), 1.0)
                    ee = max(e, 0.0)
                    sil = 1.0
                    sil2 = 1.0
                    if rim_sil:
                        sil = math.exp(-D[y, x] / (rim_px * 3.0 + 2.0))
                        sil2 = math.exp(-D[y, x] / (rim_px * 12.0 + 4.0))
                    rim = rim_amt * fr * math.exp(-ee / rim_px) * sil
                    gl = glow * frg * math.exp(-ee / (rim_px * 7.0)) * sil2
                ia = 1.0 - a
                P[y, x, 0] = v * a + P[y, x, 0] * ia
                P[y, x, 1] = rim * a + P[y, x, 1] * ia
                P[y, x, 2] = haze * a + P[y, x, 2] * ia
                P[y, x, 3] = warm * a + P[y, x, 3] * ia
                P[y, x, 4] = gl * a + P[y, x, 4] * ia
                A[y, x] = a + A[y, x] * ia


class HeadSet:
    """Ordered heads (back -> front) with their cauliflower bumps, in plate px.

    add(cx, cy, rx, ry, rng, ...) adds a head and grows its bumps; returns the head index. Keyword
    fields (defaults in HeadSet.DEF): k own-light contrast, base_y / base_soft flat-base clip, top_y
    flat-top clip, haze, tone (value offset), aa (crisp edge px), soft (lost edge px), rim / rim_px
    (sun lining), warm (warm-tint weight), beta (bump-normal weight), glow (backlit forward scatter).
    Bump controls: bump=(rmin, rmax) fraction of the head radius, arc=(a0, a1) degrees covered (screen
    angles, y down: -90 = top), sub=probability of second-scale bumps, min_px smallest bump radius."""
    DEF = dict(k=0.5, base_y=1e9, base_soft=1.0, top_y=-1e9, haze=0.0, tone=0.0, aa=0.7, soft=6.0, rim=0.0,
               rim_px=2.0, warm=0.0, beta=0.35, glow=0.0, light=(0.0, 0.0, 0.0))

    def __init__(self):
        self.H = []
        self.B = []

    def __len__(self):
        return len(self.H)

    def add(self, cx, cy, rx, ry, rng, rz=None, bump=(0.07, 0.24), arc=(-200.0, 20.0), sub=0.6, min_px=1.2,
            sink=(0.35, 0.6), spacing=1.25, sun_ang=None, sun_bias=0.4, bump_max=None, sub_scale=(0.25, 0.5),
            lumps=(1, 3), lump_r=(0.3, 0.55), clump=0.6, sub3=0.5, **kw):
        p = dict(self.DEF)
        p.update(kw)
        rz = rz if rz is not None else 0.5 * (rx + ry)
        b0 = len(self.B)
        R = 0.5 * (rx + ry)
        bmax = bump_max if bump_max is not None else 1e9
        a0r, a1r = math.radians(arc[0]), math.radians(arc[1])
        # 1) big LUMPS on the upper arc: the head is a lumpy mass, not an ellipse
        lumps_ = []
        nl = int(rng.integers(lumps[0], lumps[1] + 1)) if lumps[1] > 0 else 0
        for _ in range(nl):
            a = rng.uniform(a0r + 0.5, a1r - 0.5) if a1r - a0r > 1.2 else 0.5 * (a0r + a1r)
            if math.sin(a) > 0.2:
                a = -math.pi / 2 + (a + math.pi / 2) * 0.5
            ca, sa = math.cos(a), math.sin(a)
            rl = 1.0 / math.sqrt((ca / rx) ** 2 + (sa / ry) ** 2)
            lr = R * rng.uniform(*lump_r)
            lx = cx + ca * (rl - lr * rng.uniform(0.55, 0.85))
            ly = cy + sa * (rl - lr * rng.uniform(0.55, 0.85))
            qb = math.hypot((lx - cx) / rx, (ly - cy) / ry)
            lz = rz * math.sqrt(max(1 - qb * qb, 0.0))
            lumps_.append((lx, ly, lr, lz))

        def edge_r(a):
            # radius of the core+lumps union boundary along direction a (bisection)
            ca, sa = math.cos(a), math.sin(a)
            lo, hi = 0.0, 2.5 * R
            for _ in range(22):
                m = 0.5 * (lo + hi)
                x_, y_ = cx + ca * m, cy + sa * m
                ins = ((x_ - cx) / rx) ** 2 + ((y_ - cy) / ry) ** 2 < 1.0
                if not ins:
                    for (lx, ly, lr, lz) in lumps_:
                        if (x_ - lx) ** 2 + (y_ - ly) ** 2 < lr * lr:
                            ins = True
                            break
                if ins:
                    lo = m
                else:
                    hi = m
            return lo

        def surf_z(x_, y_):
            qb = math.hypot((x_ - cx) / rx, (y_ - cy) / ry)
            z = rz * math.sqrt(max(1 - qb * qb, 0.0))
            for (lx, ly, lr, lz) in lumps_:
                d2 = (x_ - lx) ** 2 + (y_ - ly) ** 2
                if d2 < lr * lr:
                    z = max(z, lz + math.sqrt(lr * lr - d2))
            return z

        for L_ in lumps_:
            self.B.append(L_)
        # 2) walk the union outline laying cauliflower bumps edge to edge; a slow random 'fertility'
        #    alternates dense cauliflower stretches with bare, broad ones (no bead chains)
        a = a0r + rng.uniform(0, 0.1)
        fph = rng.uniform(0, 6.28, 2)
        while a < a1r:
            fert = 0.5 + 0.5 * math.sin(a * 2.3 + fph[0]) * math.sin(a * 3.7 + fph[1])
            fert = (1 - clump) + clump * fert
            u = rng.random() ** 2.0
            fr = bump[0] + (bump[1] - bump[0]) * u
            side = 0.5 + 0.5 * math.sin(-a)                  # 1 at the top, 0.5 at the horizontal
            fr *= (0.55 + 0.45 * side) * (0.6 + 0.6 * fert)
            if sun_ang is not None:
                fr *= 1.0 + sun_bias * 0.5 * math.cos(a - sun_ang)
            ca, sa = math.cos(a), math.sin(a)
            rl = edge_r(a)
            br = min(fr * R, bmax)
            if br >= min_px and rng.random() < 0.35 + 0.65 * fert:
                sk = rng.uniform(*sink)
                bx = cx + ca * (rl - br * sk)
                by = cy + sa * (rl - br * sk)
                bz = surf_z(bx, by)
                self.B.append((bx, by, br, bz))
                # second scale: small florets on the upper / outer arc of this bump
                if br >= 3 * min_px and rng.random() < sub * fert:
                    n2 = int(rng.integers(2, 5))
                    for m in range(n2):
                        aa_ = a + rng.uniform(-1.2, 1.2)
                        aa_ = aa_ if math.sin(aa_) < 0.35 else a
                        sr = br * rng.uniform(*sub_scale)
                        if sr < min_px:
                            continue
                        c2, s2 = math.cos(aa_), math.sin(aa_)
                        sk2 = rng.uniform(0.3, 0.6)
                        sx_ = bx + c2 * (br - sr * sk2)
                        sy_ = by + s2 * (br - sr * sk2)
                        dd = math.hypot(sx_ - bx, sy_ - by)
                        sz = bz + math.sqrt(max(br * br - dd * dd, 0.0))
                        self.B.append((sx_, sy_, sr, sz))
                        # third scale: the finest cauliflower grains on the outer arc
                        if sr >= 2.5 * min_px and rng.random() < sub3:
                            for _q in range(int(rng.integers(1, 4))):
                                a3 = aa_ + rng.uniform(-1.1, 1.1)
                                a3 = a3 if math.sin(a3) < 0.4 else aa_
                                tr = sr * rng.uniform(0.3, 0.5)
                                if tr < min_px * 0.8:
                                    continue
                                c3, s3 = math.cos(a3), math.sin(a3)
                                k3 = rng.uniform(0.3, 0.6)
                                tx, ty = sx_ + c3 * (sr - tr * k3), sy_ + s3 * (sr - tr * k3)
                                d3 = math.hypot(tx - sx_, ty - sy_)
                                self.B.append((tx, ty, tr, sz + math.sqrt(max(sr * sr - d3 * d3, 0.0))))
            br = max(br, min_px)
            step = br * spacing * rng.uniform(0.8, 1.3) / max(rl, 1e-6)
            a += max(step, 0.02)
        # fillers: a chain of circles from every bump back toward the head centre, so no bump can bridge a
        # notch and enclose a pocket (the head is star-shaped); fillers never win the depth test
        fill = []
        for (bx, by, br, bz) in self.B[b0:]:
            dx_, dy_ = cx - bx, cy - by
            d_ = math.hypot(dx_, dy_)
            if d_ < 1e-6:
                continue
            ux_, uy_ = dx_ / d_, dy_ / d_
            px_, py_ = bx, by
            for _ in range(12):
                if ((px_ - cx) / rx) ** 2 + ((py_ - cy) / ry) ** 2 < 0.95:
                    break
                inside = False
                for (lx, ly, lr, lz) in lumps_:
                    if (px_ - lx) ** 2 + (py_ - ly) ** 2 < (lr * 0.95) ** 2:
                        inside = True
                        break
                if inside:
                    break
                px_, py_ = px_ + ux_ * br * 0.8, py_ + uy_ * br * 0.8
                fill.append((px_, py_, br, -1e9))
        self.B.extend(fill)
        nb = len(self.B) - b0
        pad = p['soft'] * 2 + 3
        if nb:
            Bs = np.asarray(self.B[b0:], np.float64)
            x0 = min(cx - rx, float((Bs[:, 0] - Bs[:, 2]).min())) - pad
            x1 = max(cx + rx, float((Bs[:, 0] + Bs[:, 2]).max())) + pad
            y0 = min(cy - ry, float((Bs[:, 1] - Bs[:, 2]).min())) - pad
            y1 = max(cy + ry, float((Bs[:, 1] + Bs[:, 2]).max())) + pad
        else:
            x0, x1, y0, y1 = cx - rx - pad, cx + rx + pad, cy - ry - pad, cy + ry + pad
        if p['base_y'] < 1e8:
            y1 = min(y1, p['base_y'] + p['base_soft'] * 2 + 2)
        self.H.append((cx, cy, rx, ry, rz, p['k'], p['base_y'], p['base_soft'], p['top_y'], p['haze'], p['tone'],
                       p['aa'], p['soft'], p['rim'], p['rim_px'], p['warm'], b0, nb, x0, y0, x1, y1, p['beta'],
                       p['glow'], float(p['light'][0]), float(p['light'][1]), float(p['light'][2])))
        return len(self.H) - 1

    def arrays(self):
        Hs = np.asarray(self.H, np.float64).reshape(-1, _HCOLS)
        Bs = np.asarray(self.B, np.float64).reshape(-1, 4)
        if len(Bs) == 0:
            Bs = np.zeros((1, 4))
        return Hs, Bs


def _light_vec(sun_dir, sun_z):
    d = np.asarray(sun_dir, np.float64)
    d = d / (np.linalg.norm(d) + 1e-9)
    v = np.array([d[0], d[1], sun_z], np.float64)          # +z = toward the viewer
    return v / np.linalg.norm(v)


def form_field(U, L, sigma, shadow_len, dens=2.5, wrap=0.4, base_y=None, base_h=None, base_dark=0.35,
               floor=0.25, ds=4):
    """Big-form light (0..1) of a union silhouette U (H, W): blurred-mask normal x a screen-space
    sun march (upper / sunward mass shades what is under / behind it), darker toward base_y.
    Returns (V, T) where T is the cast-light term alone."""
    h, w = U.shape
    s = max(1, int(ds))
    u = _resize(U, max(w // s, 2), max(h // s, 2), cv2.INTER_AREA)
    sg = max(sigma / s, 1.0)
    ub = cv2.GaussianBlur(u, (0, 0), sg)
    gx = cv2.Sobel(ub, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(ub, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    kk = sg * 2.5
    nx, ny, nz = -gx * kk, -gy * kk, np.clip(ub, 0.08, 1.0) * 0.9 + 0.1
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    lam = (nx * L[0] + ny * L[1] + nz * L[2]) / nn
    lam = np.clip((lam + wrap) / (1 + wrap), 0, 1)
    l2 = np.array([L[0], L[1]], np.float64)
    l2 = l2 / (np.linalg.norm(l2) + 1e-9)
    n = 20
    stp = shadow_len / s / n
    ub2 = cv2.GaussianBlur(u, (0, 0), max(sg * 0.35, 1.0))
    acc = np.zeros_like(u)
    for i in range(1, n + 1):
        acc += _shift_img(ub2, -l2[0] * stp * i, -l2[1] * stp * i)
    T = np.exp(-dens * acc / n * (1.0 - 0.6 * max(L[2], 0.0)))
    T = cv2.GaussianBlur(T, (0, 0), max(sg * 0.4, 1.0))
    V = lam * (floor + (1 - floor) * T)
    if base_y is not None:
        ys = (np.arange(u.shape[0], dtype=F32)[:, None] + 0.5) * s
        rows = np.nonzero(u.max(1) > 0.5)[0]
        top = float(rows.min() * s) if len(rows) else 0.0
        bh = base_h if base_h is not None else 0.3 * (base_y - top)
        V = V * (1 - base_dark * (1 - _ss(0.0, bh, base_y - ys)))
    V = _resize(V.astype(F32), w, h, cv2.INTER_LINEAR)
    T = _resize(T.astype(F32), w, h, cv2.INTER_LINEAR)
    return V, T


RAMPS = {
    # value stops 0..1 -> colour; the warm ramp is mixed in on the sunlit flank
    'noon': dict(stops=[(0.0, (0.3, 0.46, 0.72)), (0.22, (0.39, 0.56, 0.8)), (0.42, (0.54, 0.7, 0.87)),
                        (0.6, (0.75, 0.85, 0.93)), (0.78, (0.93, 0.955, 0.96)), (1.0, (1.04, 1.02, 0.97))],
                 warm=[(0.0, (0.46, 0.5, 0.74)), (0.42, (0.7, 0.72, 0.86)), (0.6, (0.9, 0.88, 0.88)),
                       (0.78, (1.0, 0.96, 0.88)), (1.0, (1.06, 1.01, 0.88))],
                 rim=(1.25, 1.2, 1.08), haze=(0.72, 0.84, 0.96), glow=(1.1, 1.0, 0.85)),
    'sunrise': dict(stops=[(0.0, (0.07, 0.07, 0.21)), (0.25, (0.15, 0.14, 0.4)), (0.45, (0.27, 0.24, 0.58)),
                           (0.65, (0.46, 0.36, 0.7)), (0.85, (0.84, 0.54, 0.66)), (1.0, (1.1, 0.76, 0.6))],
                    warm=[(0.0, (0.14, 0.1, 0.28)), (0.4, (0.42, 0.28, 0.5)), (0.7, (0.9, 0.52, 0.5)),
                          (1.0, (1.15, 0.8, 0.52))],
                    rim=(1.3, 0.76, 0.36), haze=(0.98, 0.78, 0.7), glow=(1.25, 0.66, 0.4)),
}


def _ramp(stops, v):
    xs = np.array([s[0] for s in stops], F32)
    cs = np.array([s[1] for s in stops], F32)
    return np.stack([np.interp(v, xs, cs[:, c]) for c in range(3)], -1).astype(F32)


def paint_heads(w, h, hs, sun_dir=(0.75, -0.65), sun_z=0.3, ramp='noon', wrap=0.35, form_sigma=None,
                shadow_len=None, dens=2.5, base_y=None, base_h=None, base_dark=0.35, floor=0.3, Vbig=None,
                steps=((0.3, 1.0), (0.52, 1.0), (0.72, 1.0)), step_soft=0.035, poster=0.55, brush=0.035,
                warp=0.06, kuwa=0.0, seed=0, lost_up=0.0, warm_split=0.7, haze_col=None, crop=True,
                return_fields=False, detail_depth=None, k_inside=0.6, crust=0.35, crust_mix=0.65, soft_inside=0.35,
                edge_noise=1.6, smooth_k=None, close_px=4.0, rim_sil=True, rim_replace=0.5, rim_up=0.0,
                crust_bthr=0.0):
    """Paint a HeadSet into a straight-alpha RGBA plate (h, w, 4).

    sun_dir (screen, y down) + sun_z (+ toward viewer; < 0 = backlit); ramp: RAMPS key or dict;
    form_sigma / shadow_len: big-form blur and sun-march length (px, default from the set's size);
    dens: self-shadow density; base_y / base_h / base_dark: darken toward the base; floor: min light in
    cast shadow; Vbig: optional precomputed big-form field (h, w) (skips the form pass); steps: painted
    value planes (threshold, weight); step_soft: border softness; poster: 0 smooth .. 1 flat planes;
    brush: stroke texture; warp: border wobble; kuwa: Kuwahara radius (px, 0 = off); warm_split: warm
    cream on the lit / sunward flank vs cool shade; lost_up: also lose up-facing edges (backlit)."""
    R = RAMPS[ramp] if isinstance(ramp, str) else ramp
    L = _light_vec(sun_dir, sun_z)
    Hs, Bs = hs.arrays()
    out = np.zeros((h, w, 4), F32)
    if len(Hs) == 0:
        return (out, {}) if return_fields else out
    if crop:
        x0, y0 = int(max(Hs[:, 18].min(), 0)), int(max(Hs[:, 19].min(), 0))
        x1, y1 = int(min(Hs[:, 20].max() + 1, w)), int(min(Hs[:, 21].max() + 1, h))
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    if x1 <= x0 or y1 <= y0:
        return (out, {}) if return_fields else out
    ww, hh = x1 - x0, y1 - y0
    H2 = Hs.copy()
    H2[:, [0, 18, 20]] -= x0
    H2[:, [1, 19, 21]] -= y0
    H2[:, 6] = np.where(H2[:, 6] < 1e8, H2[:, 6] - y0, H2[:, 6])
    H2[:, 8] = np.where(H2[:, 8] > -1e8, H2[:, 8] - y0, H2[:, 8])
    B2 = Bs.copy()
    B2[:, 0] -= x0
    B2[:, 1] -= y0
    size = max(float(np.ptp(H2[:, 0]) + 2 * H2[:, 2].max()), float(np.ptp(H2[:, 1]) + 2 * H2[:, 3].max()))
    sc = max(w / 1920.0, 0.25)
    smooth_k = float(smooth_k if smooth_k is not None else 6.0 * sc)
    # brush-dab edge noise: breaks perfect circles into painted, slightly ragged edges
    namp = float(edge_noise) * sc
    if namp > 0:
        Nz = (0.65 * _noise(ww, hh, max(ww / (16.0 * sc), 4), seed + 21, 2, stretch=2.0, angle=-30) +
              0.35 * _noise(ww, hh, max(ww / (5.0 * sc), 8), seed + 22, 2)).astype(F32)
    else:
        Nz = np.zeros((1, 1), F32)
    T = None
    if Vbig is None:
        U = np.zeros((hh, ww), F32)
        Pd = np.zeros((1, 1, 5), F32)
        _paint_heads_k(Pd, U, np.zeros((1, 1), F32), np.zeros((1, 1), F32), H2, B2, L[0], L[1], L[2], wrap, True,
                       lost_up, 1.0, 1.0, 0.3, 0.0, 0.0, Nz, namp, smooth_k, True, 0.0, 0.0)
        fs = form_sigma if form_sigma is not None else 0.06 * size
        sl = shadow_len if shadow_len is not None else 0.35 * size
        by = None if base_y is None else base_y - y0
        Vb, T = form_field(U, L, fs, sl, dens=dens, wrap=wrap, base_y=by, base_h=base_h, base_dark=base_dark,
                           floor=floor)
    else:
        Vb = np.ascontiguousarray(Vbig[y0:y1, x0:x1]).astype(F32)
        U = np.zeros((hh, ww), F32)
        _paint_heads_k(np.zeros((1, 1, 5), F32), U, Vb, np.zeros((1, 1), F32), H2, B2, L[0], L[1], L[2], wrap, True,
                       lost_up, 1.0, 1.0, 0.3, 0.0, 0.0, Nz, namp, smooth_k, True, 0.0, 0.0)
    Dm = cv2.distanceTransform((U > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    ds_ = detail_depth if detail_depth is not None else 0.03 * size
    P = np.zeros((hh, ww, 5), F32)
    A = np.zeros((hh, ww), F32)
    _paint_heads_k(P, A, Vb, Dm, H2, B2, L[0], L[1], L[2], wrap, False, lost_up, max(ds_, 1.0), k_inside,
                   crust, crust_mix, soft_inside, Nz, namp, smooth_k, rim_sil, rim_up, crust_bthr)
    ia = 1.0 / np.maximum(A, 1e-4)
    V = P[..., 0] * ia
    RIM = P[..., 1] * ia
    HZ = P[..., 2] * ia
    WM = P[..., 3] * ia
    GL = P[..., 4] * ia
    if _DEBUG is not None:
        _DEBUG['fields'] = dict(V=V.copy(), RIM=RIM.copy(), GL=GL.copy(), HZ=HZ.copy(), WM=WM.copy(), A=A.copy(),
                                x0=x0, y0=y0)
    if close_px:
        # grey closing removes small dark pits (notch pockets, tiny heads' undersides) - no AO dimples
        r_ = max(int(round(close_px * sc)), 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))
        Vc = cv2.morphologyEx(V.astype(F32), cv2.MORPH_CLOSE, ker)
        inner = cv2.erode((A > 0.98).astype(np.uint8), ker).astype(F32)
        V = V + (Vc - V) * inner
    if brush:
        bt = _noise(ww, hh, max(ww / (22.0 * sc), 4), seed + 11, 3, stretch=3.0, angle=-25)
        V = V + brush * bt
    if poster:
        wn = _noise(ww, hh, max(ww / (60.0 * sc), 3), seed + 12, 3) * warp
        Vq = np.zeros_like(V)
        tot = 0.0
        for (t_, wt) in steps:
            Vq += wt * _ss(t_ - step_soft, t_ + step_soft, V + wn)
            tot += wt
        Vq = Vq / max(tot, 1e-6)
        # stepped planes, each keeping a little of the underlying gradient (painted, not cel)
        V = V * (1 - poster) + (0.1 + 0.85 * Vq + 0.15 * (V - 0.5)) * poster
    V = np.clip(V, 0, 1.2)
    cool = _ramp(R['stops'], V)
    warmc = _ramp(R['warm'], V)
    lit_side = _ss(0.35, 0.85, T) if T is not None else np.ones_like(V)
    wk = np.clip(WM * warm_split * lit_side * _ss(0.35, 0.75, V), 0, 1)
    col = cool * (1 - wk[..., None]) + warmc * wk[..., None]
    if kuwa:
        col = kuwahara(col, kuwa, q=5.0)
    # linings go on AFTER the painterly flattening (thin lines survive)
    rr = np.clip(RIM, 0, 1.5)[..., None]
    col = col * (1 - np.clip(rr, 0, 1) * rim_replace) + np.asarray(R['rim'], F32) * rr * 0.9
    col = col + np.asarray(R['glow'], F32) * np.clip(GL, 0, 1)[..., None] * 0.35
    hc = np.asarray(haze_col if haze_col is not None else R['haze'], F32)
    col = col * (1 - HZ[..., None]) + hc * HZ[..., None]
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = np.clip(A, 0, 1)
    out[..., :3] = _bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    if return_fields:
        return out, dict(V=V, T=T, U=A, x0=x0, y0=y0)
    return out


def _tower_profile(s, kind='cb'):
    """Half-width (fraction of the half width) at normalised height s (0 base .. 1 top)."""
    s = float(np.clip(s, 0, 1))
    if kind == 'heap':
        return (0.8 + 0.2 * math.sin(math.pi * min(s * 1.4, 1.0))) * math.sqrt(max(1 - s ** 2.4, 0.0))
    w = 0.98 - 0.4 * s
    cap = math.sqrt(max(1 - max((s - 0.8) / 0.2, 0.0) ** 2, 0.0))
    return w * (0.4 + 0.6 * cap)


def tower_heads(hs, rng, cx, base_y, height, width, lean=0.0, sun_dir=(0.75, -0.65), kind='cb', sc=1.0,
                haze=0.0, row_k=0.62, children=(2.6, 1.6), child_r=((0.3, 0.52), (0.3, 0.5)),
                k_levels=(0.7, 0.45, 0.3), soft=0.04, aa=0.7, rim=0.12, rim_px=1.4, glow=0.0, base_soft=2.0,
                bump=(0.07, 0.22), wobble=0.1, beta=0.35, tone=0.0, side_heads=1.0, base_flat=0.6,
                min_px=1.1, top_y=None, sub=0.6, bump_px=None, edge_keep=(0.25, 1.5), mass_density=1.3,
                mass_r=(0.35, 0.72)):
    """Lay out one towering cumulus / cumulonimbus (or a heap with kind='heap') as stacked heads.

    Rows of big MASSES (level 0) are stacked from the crown down to the flat base, each row a little in
    front of the one above; on the upper / sunward arc of every mass sit a few medium HEADS (level 1)
    and on those a few small FLORETS (level 2), so detail concentrates on the silhouette and the lit
    crowns. Every head carries its own cauliflower bumps. sc = px scale (plate width / 1920)."""
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    sun_ang = math.atan2(sd[1], sd[0])
    hw = width * 0.5
    ph = rng.uniform(0, 6.28, 3)

    def axis_x(s):
        return cx + lean * height * s + wobble * hw * (0.6 * math.sin(2.2 * s * math.pi + ph[0]) +
                                                        0.4 * math.sin(4.3 * s * math.pi + ph[1]))

    bpx = bump_px if bump_px is not None else 1e9
    common = dict(base_y=base_y, base_soft=base_soft * sc, haze=haze, aa=aa * sc, rim=rim, rim_px=rim_px * sc,
                  glow=glow, warm=1.0, beta=beta, min_px=min_px * sc, sun_ang=sun_ang, sub=sub, bump_max=bpx * sc)
    if top_y is not None:
        common['top_y'] = top_y

    def add_head(x, y, r, level, flat=0.85, hz=0.0, tn=0.0):
        rx = r * rng.uniform(1.0, 1.18)
        ry = r * flat * rng.uniform(0.9, 1.05)
        arc = (-215.0, 55.0) if level == 0 else (-205.0, 40.0)
        return hs.add(x, y, rx, ry, rng, k=k_levels[level], soft=max(soft * r, 1.5 * sc), bump=bump, arc=arc,
                      tone=tone + tn, **dict(common, haze=min(haze + hz, 1.0)))

    def grow(x, y, rx, ry, level, n_mean):
        if level > 2:
            return
        n = rng.poisson(n_mean)
        for _ in range(n):
            # upper arc, biased toward the sun
            a = rng.uniform(-170, -10)
            a = math.radians(a)
            if rng.random() < 0.6:
                a = a * 0.6 + 0.4 * (sun_ang if math.sin(sun_ang) < 0 else -math.pi / 2)
            lo, hi = child_r[level - 1]
            r = 0.5 * (rx + ry) * rng.uniform(lo, hi)
            if r < 3 * sc:
                continue
            ca, sa = math.cos(a), math.sin(a)
            rl = 1.0 / math.sqrt((ca / rx) ** 2 + (sa / ry) ** 2)
            px_ = x + ca * (rl - r * rng.uniform(0.45, 0.75))
            py_ = y + sa * (rl - r * rng.uniform(0.45, 0.75))
            if py_ + r * 0.3 > base_y:
                continue
            # medium / small heads concentrate toward the envelope edge and the crown (calm interior)
            s_ = float(np.clip((base_y - py_) / height, 0, 1))
            pw_ = max(_tower_profile(s_, kind) * hw, 1.0)
            edge = min(abs(px_ - axis_x(s_)) / pw_, 1.0)
            if rng.random() > edge_keep[0] + (1 - edge_keep[0]) * max(edge, _ss(0.8, 1.0, s_)) ** edge_keep[1]:
                continue
            i = add_head(px_, py_, r, level, flat=rng.uniform(0.78, 0.92))
            h_ = hs.H[i]
            grow(px_, py_, h_[2], h_[3], level + 1, children[1] if level == 1 else 0)

    # big MASSES scattered through the envelope (power-law sizes), painted crown -> base with a depth
    # jitter, so the column is a pile of overlapping heads (no rows / pancake tiers). A spine of axis
    # masses guarantees a filled column.
    items = []
    area = 0.0
    for s_ in np.linspace(0, 1, 40):
        area += 2 * _tower_profile(s_, kind) * hw * height / 40
    n0 = int(area / (math.pi * (0.42 * hw) ** 2) * mass_density) + 3
    for m in range(n0):
        s = rng.random() ** 0.9
        pw = _tower_profile(s, kind) * hw
        if pw < 0.05 * hw:
            continue
        r = pw * (mass_r[0] + (mass_r[1] - mass_r[0]) * rng.random() ** 1.6)
        x = axis_x(s) + rng.uniform(-1, 1) * max(pw - r * 0.8, 0.0)
        y = base_y - s * height + r * 0.6
        items.append((y + rng.uniform(-0.35, 0.35) * r, x, y, r, 0))
    nsp = max(int(height / (0.55 * hw)), 3)
    for m in range(nsp):
        s = (m + 0.5) / nsp
        pw = _tower_profile(s, kind) * hw
        r = pw * rng.uniform(0.62, 0.8)
        y = base_y - s * height + r * 0.5
        items.append((y - 0.6 * r, axis_x(s) + rng.uniform(-0.15, 0.15) * pw, y, r, 0))
    # base: a few wide flat masses sitting on the base plane
    pw0 = _tower_profile(0.0, kind) * hw
    nb_ = max(int(round(2 * pw0 / (0.9 * pw0))), 2)
    for m in range(nb_):
        u = (m + 0.5) / nb_ * 2 - 1
        r = pw0 * rng.uniform(0.5, 0.65)
        x = axis_x(0.0) + u * (pw0 - r * 0.8)
        items.append((base_y + 1e5 + m, x, base_y - r * base_flat * 0.55, r, 1))
    items.sort(key=lambda t: t[0])
    for (_, x, y, r, isb) in items:
        i = add_head(x, y, r, 0, flat=base_flat if isb else rng.uniform(0.78, 0.95))
        h_ = hs.H[i]
        grow(x, y, h_[2], h_[3], 1, children[0] * (0.5 if isb else 1.0))
    return hs


def union_alpha(w, h, hs, lost_up=0.0):
    """Coverage (h, w) of a HeadSet (crisp pass, no shading)."""
    Hs, Bs = hs.arrays()
    U = np.zeros((h, w), F32)
    if len(Hs):
        _paint_heads_k(np.zeros((1, 1, 5), F32), U, np.zeros((1, 1), F32), np.zeros((1, 1), F32), Hs, Bs, 0.6, -0.6,
                       0.5, 0.3, True, lost_up, 1.0, 1.0, 0.3, 0.0, 0.0, np.zeros((1, 1), F32), 0.0, 6.0 * w / 1920.0, True,
                       0.0, 0.0)
    return U


def add_fringe(hs, w, h, rng, sun_dir=(0.75, -0.65), r=(2.0, 9.0), density=1.0, up_min=0.05, sun_w=0.75, sc=1.0,
               sink=(0.6, 0.9), clump=0.6, region=None, **kw):
    """Fine cauliflower FRINGE on the union silhouette: walks the outer contour of the set and adds tiny
    heads (no lumps / bumps of their own) where the outline faces up or toward the sun, in clumps, so the
    lit contour gets the finest scale of detail while the shadow / down side stays smooth. r = floret
    radius range at 1080p (px, scaled by sc); kw -> HeadSet.add (k, base_y, haze, aa, soft, rim, ...)."""
    U = union_alpha(w, h, hs)
    m = (U > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    ub = cv2.GaussianBlur(U, (0, 0), 3.0 * sc + 1.0)
    gy, gx = np.gradient(ub)
    ph = rng.uniform(0, 6.28, 3)
    kw.setdefault('lumps', (0, 0))
    kw.setdefault('bump', (0.0, 0.0))
    kw.setdefault('arc', (0.0, 0.0))
    for c in cs:
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) < 20:
            continue
        # arc length parametrisation
        seg = np.sqrt((np.diff(pts, axis=0, append=pts[:1]) ** 2).sum(1))
        sacc = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
        total = float(seg.sum())
        pos = rng.uniform(0, 3.0)
        while pos < total:
            i = int(np.searchsorted(sacc, pos))
            i = min(i, len(pts) - 1)
            x, y = pts[i]
            xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
            nx, ny = -gx[yi, xi], -gy[yi, xi]
            nn = math.hypot(nx, ny) + 1e-9
            nx, ny = nx / nn, ny / nn
            up = -ny
            face = nx * sd[0] + ny * sd[1]
            fert = 0.5 + 0.5 * math.sin(pos / (60.0 * sc) + ph[0]) * math.sin(pos / (23.0 * sc) + ph[1])
            fert = (1 - clump) + clump * fert
            ok = (up > up_min) and (region is None or region(x, y))
            wgt = np.clip((up - up_min) / (1 - up_min), 0, 1) * (1 - sun_w + sun_w * max(face, 0.0))
            rr = (r[0] + (r[1] - r[0]) * rng.random() ** 2.2) * sc * (0.6 + 0.6 * fert)
            if ok and rng.random() < density * wgt * fert * 1.6:
                sk = rng.uniform(*sink)
                hs.add(x - nx * rr * sk, y - ny * rr * sk, rr * rng.uniform(1.0, 1.25), rr * rng.uniform(0.85, 1.0),
                       rng, **kw)
            pos += rr * rng.uniform(0.9, 1.6)
    return hs


def anvil_heads(hs, rng, xt, top_y, left, right, thick, sc=1.0, k=0.45, haze=0.0, aa=0.7, soft=0.06, rim=0.12,
                rim_px=1.4, glow=0.0, warm=1.0, beta=0.3, sun_dir=(0.75, -0.65), fringe_len=0.0, light=(0, 0, 0),
                tone=0.0, flare=0.0):
    """Flat-topped anvil (wwy_02) in the same head language: a sheet of wide flat heads cut by a flat top
    plane (top_y), thickest over the column (xt) and thinning to a long blade downwind (`right` / `left` =
    extents in px), a blunt upwind end, a scalloped cauliflower UNDERSIDE (bumps on the lower arc) and a
    small frayed lip along the top edge. Paint it BEFORE the column heads (the crown sits in front of the
    anvil's shaded underside)."""
    sd = np.asarray(sun_dir, np.float64)
    sun_ang = math.atan2(sd[1], sd[0])
    down_right = right >= left
    span = left + right
    n = max(int(span / (thick * 0.9)), 4)
    xs = np.linspace(xt - left, xt + right, n)
    order = np.argsort(np.abs(xs - xt))[::-1]              # outer blades first, the core last (in front)
    for i in order:
        x = xs[i] + rng.uniform(-0.2, 0.2) * thick
        u = (x - xt) / (right if x >= xt else left)
        downwind = (u > 0) == down_right
        au = min(abs(u), 1.0)
        t = thick * ((1 - 0.6 * au ** 1.3) if downwind else (1 - 0.45 * au ** 1.5))
        rx = thick * rng.uniform(1.1, 1.6) * (1 - 0.4 * au)
        ry = max(t * rng.uniform(0.6, 0.8), 2.0 * sc)
        cy = top_y + ry * rng.uniform(0.5, 0.8)
        hs.add(x, cy, rx, ry, rng, k=k, top_y=top_y + rng.uniform(-0.03, 0.03) * thick, haze=haze, aa=aa * sc,
               soft=max(soft * ry, 1.5 * sc), rim=rim, rim_px=rim_px * sc, glow=glow, warm=warm, beta=beta,
               bump=(0.12, 0.35), arc=(-10.0, 190.0), lumps=(0, 2), lump_r=(0.25, 0.45), sub=0.5, min_px=1.1 * sc,
               sun_ang=sun_ang, light=light, tone=tone)
    # flare: the column spreading up into the anvil (a cone of heads under the sheet, widening upward)
    if flare:
        col_hw = flare
        nf = 10
        for m in range(nf):
            s_ = m / (nf - 1)                                  # 0 = low on the column .. 1 = under the sheet
            y = top_y + thick * (0.9 + 2.6 * (1 - s_))
            half = col_hw * (1 + 1.6 * s_ ** 1.5)
            for side in (-1, 1):
                ext = right if side > 0 else left
                x = xt + side * min(half, ext * 0.8) * rng.uniform(0.45, 0.8)
                r = thick * rng.uniform(0.55, 0.85) * (0.8 + 0.5 * s_)
                hs.add(x, y, r * 1.25, r * 0.8, rng, k=k, haze=haze, aa=aa * sc, soft=max(soft * r, 1.5 * sc),
                       rim=rim, rim_px=rim_px * sc, glow=glow, warm=warm, beta=beta, bump=(0.1, 0.3),
                       arc=(-200.0, 20.0), lumps=(0, 2), sub=0.5, min_px=1.1 * sc, sun_ang=sun_ang, light=light,
                       tone=tone)
    # frayed lip: a line of small florets along the flat top edge
    if fringe_len:
        x = xt - left
        while x < xt + right:
            u = (x - xt) / (right if x >= xt else left)
            r = thick * rng.uniform(0.06, 0.16) * (1 - 0.5 * min(abs(u), 1.0))
            if r > 1.2 * sc:
                hs.add(x, top_y + r * 0.3, r * 1.4, r * 0.7, rng, k=k * 0.8, haze=haze, aa=aa * sc, soft=1.5 * sc,
                       rim=rim, rim_px=rim_px * sc, glow=glow, warm=warm, beta=beta, lumps=(0, 0), bump=(0, 0),
                       arc=(0, 0), light=light, tone=tone)
            x += max(r * rng.uniform(1.2, 3.5), 1.5 * sc)
    return hs


def cloud_tower_plate(w, h, cx, base_y, height, width, sun_dir=(0.75, -0.65), sun_z=0.95, ramp='noon', seed=0,
                      lean=0.1, kind='cb', anvil=None, fringe=1.0, haze=0.0, tower=None, fr=None, light=(0, 0, 0),
                      extra=None, **paint):
    """One towering cumulus / cumulonimbus (or a heap, kind='heap') as a finished straight-alpha RGBA plate,
    painted with the heads engine (paint_heads). anvil: None or dict(left, right, thick[, top]) px;
    fringe: silhouette floret density (0 = off); tower: extra kwargs for tower_heads; fr: extra kwargs
    for add_fringe; extra: optional callable(hs, rng) adding more heads before the fringe pass;
    **paint -> paint_heads (sun_z, dens, floor, crust, poster, brush, kuwa, ...)."""
    rng = np.random.default_rng(seed)
    sc = w / 1920.0 / 1.0
    hs = HeadSet()
    tw = dict(tower or {})
    if anvil is not None:
        a = dict(anvil)
        top = a.pop('top', base_y - height)
        a.setdefault('flare', 0.5 * width * _tower_profile(0.9, kind))
        anvil_heads(hs, rng, cx + lean * height, top, a.pop('left'), a.pop('right'), a.pop('thick'), sc=sc,
                    haze=haze, sun_dir=sun_dir, light=light, **a)
    tower_heads(hs, rng, cx, base_y, height, width, lean=lean, sun_dir=sun_dir, kind=kind, sc=sc, haze=haze, **tw)
    if extra is not None:
        extra(hs, rng)
    if fringe:
        f = dict(k=0.15, base_y=base_y, warm=1.0, aa=0.7 * sc, soft=1.5 * sc, beta=0.1, haze=haze, density=fringe,
                 light=light)
        f.update(fr or {})
        add_fringe(hs, w, h, rng, sun_dir=sun_dir, sc=sc, **f)
    paint.setdefault('base_y', base_y)
    return paint_heads(w, h, hs, sun_dir=sun_dir, sun_z=sun_z, ramp=ramp, seed=seed, **paint)


def mackerel_plate(w, h, horizon_y, seed=0, region=(0.0, 0.55), angle=-20.0, size=(34.0, 6.0), density=0.8,
                   band=0.09, band_fill=0.55, patch=0.45, avoid=(), sun_dir=(0.75, -0.65), sun_z=0.6, ramp='noon',
                   value=0.78, haze_top=0.0, haze_bottom=0.55, opacity=0.95, aspect=(0.2, 0.34), veil=0.25,
                   veil_col=None, kuwa=2, k=0.75, bump=(0.18, 0.4), light=(0, 0, 0), warp=0.6):
    """Altocumulus / mackerel sky (gow_02, yn_02) in the heads language: small elongated cloudlets, each a
    tiny head with its own cauliflower bumps, a crisp crust-lit top and a soft, lost underside, laid along
    STREAMING BANDS (angle = flow direction, degrees, screen) that wander (warp) and break into patches.
    Size falls off with perspective from size[0] (px at 1080p, top of region) to size[1] (near the horizon)
    and the spacing scales with it, so the field recedes. avoid: [(x, y, rx, ry)] ellipses (plate px) where
    the field thins out (clean sky around a hero cloud). veil: soft airbrushed glow under dense patches."""
    rng = np.random.default_rng(seed)
    sc = w / 1920.0
    y0, y1 = region[0] * h, min(region[1] * h, horizon_y - 2)
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)                   # along the flow (screen, y down)
    vx, vy = -uy, ux
    cells = max(w / (patch * w + 1), 1.0)
    pn = _noise(w, h, max(w / (patch * w), 1.5), seed + 3, 3, stretch=2.0, angle=-angle)
    wn = _noise(w, h, max(w / (0.25 * w), 2.0), seed + 4, 2)
    pn = np.clip((pn - np.percentile(pn, 15)) / (np.percentile(pn, 90) - np.percentile(pn, 15) + 1e-6), 0, 1)
    hs = HeadSet()
    sd = np.asarray(sun_dir, np.float64)
    sun_ang = math.atan2(sd[1], sd[0])
    items = []
    # walk rows perpendicular to v (i.e. along the flow) across a rotated grid covering the region
    D = math.hypot(w, h)
    cxs, cys = w / 2, (y0 + y1) / 2
    vpos = -D
    while vpos < D:
        # perspective size at this row's centre line
        upos = -D
        while upos < D:
            x = cxs + upos * ux + vpos * vx
            y = cys + upos * uy + vpos * vy
            kk = float(np.clip((y - y0) / max(horizon_y - y0, 1), 0, 1))
            s_ = (size[0] + (size[1] - size[0]) * kk ** 0.8) * sc
            step_u = s_ * rng.uniform(1.4, 2.4)
            r1, r2, r3, r4 = rng.random(), rng.random(), rng.random(), rng.random()
            if 0 <= x < w and y0 - s_ < y < y1:
                xi, yi = int(x), int(min(max(y, 0), h - 1))
                # streaming bands: a wavy stripe pattern across the flow, broken by the patch mask
                ph = (vpos + warp * band * w * wn[yi, xi]) / (band * w * (0.5 + 0.8 * (1 - kk)))
                stripe = 0.5 + 0.5 * math.cos(ph * 2 * math.pi)
                pr = float(_ss(1 - band_fill, 1.0, stripe)) ** 0.6 * float(_ss(0.0, 0.6, pn[yi, xi])) * density * 1.6
                pr *= float(_ss(y0 - 0.02 * h, y0 + 0.06 * h, y)) * float(1 - _ss(y1 - 0.08 * h, y1, y))
                for (ax, ay, rx_, ry_) in avoid:
                    d2 = ((x - ax) / rx_) ** 2 + ((y - ay) / ry_) ** 2
                    pr *= float(np.clip(d2 - 0.6, 0, 1))
                if r1 < pr:
                    rx = s_ * (0.55 + 0.75 * r2 ** 1.5) * (0.7 + 0.5 * pr)
                    ry = rx * (aspect[0] + (aspect[1] - aspect[0]) * r3) * (0.75 + 0.25 * (1 - kk))
                    items.append((y, x + (r4 - 0.5) * s_ * 0.6, y, rx, ry, kk))
            upos += step_u
        kk = float(np.clip((cys + vpos * vy - y0) / max(horizon_y - y0, 1), 0, 1))
        vpos += (size[0] + (size[1] - size[0]) * kk ** 0.8) * sc * rng.uniform(0.9, 1.4)
    items.sort(key=lambda t: t[0])
    for (_, x, y, rx, ry, kk) in items:
        hz = min(haze_top + (haze_bottom - haze_top) * kk + 0.35 * rng.random() ** 3, 0.9)
        hs.add(x, y, rx, ry, rng, k=k, haze=hz, aa=0.6 * sc, soft=max(0.45 * ry, 1.0 * sc), bump=bump,
               arc=(-190.0, 10.0), lumps=(1, 3), lump_r=(0.3, 0.5), sub=0.3, sub3=0.0, min_px=0.8 * sc,
               sun_ang=sun_ang, warm=1.0, beta=0.3, light=light, tone=rng.uniform(-0.12, 0.08))
    if len(hs) == 0:
        return np.zeros((h, w, 4), F32)
    Vb = np.full((h, w), value, F32)
    out = paint_heads(w, h, hs, sun_dir=sun_dir, sun_z=sun_z, ramp=ramp, Vbig=Vb, crust=0.5, crust_mix=0.8,
                      poster=0.35, brush=0.03, kuwa=kuwa, seed=seed, k_inside=1.0, detail_depth=1e6,
                      soft_inside=0.0, edge_noise=0.8, warm_split=0.4)
    if veil:
        A = out[..., 3]
        vl = _blur(A, 12.0 * sc) * veil
        vc = np.asarray(veil_col if veil_col is not None else RAMPS[ramp if isinstance(ramp, str) else 'noon']['stops'][3][1], F32)
        a2 = np.clip(A + vl * (1 - A), 0, 1)
        out[..., :3] = (out[..., :3] * A[..., None] + vc * (vl * (1 - A))[..., None]) / np.maximum(a2, 1e-4)[..., None]
        out[..., 3] = a2
    out[..., 3] *= opacity
    out[..., :3] = _bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), 3.0)
    return out


def flecks_plate(w, h, horizon_y, seed=0, region=(0.0, 0.6), streams=6, angle=-22.0, dab_angle=-30.0,
                 size=(36.0, 5.0), per_stream=1400, width=(0.02, 0.055), wander=0.05, avoid=(), sun_dir=(0.75, -0.65),
                 lit=(1.0, 0.98, 0.92), body=(0.8, 0.88, 0.97), under=(0.55, 0.7, 0.9), veil=0.18,
                 veil_col=(0.75, 0.86, 0.97), opacity=0.92, haze_col=None, haze_bottom=0.6, ss=2, x_range=(-0.1, 1.1),
                 y_bias=None, dry=1.0, clumps=1.0):
    """Altocumulus / mackerel flecks (gow_02, yn_02): torn, sheared dabs packed in STREAMS.

    `streams` curved stream lines cross the region along `angle` (deg, screen) with a slow wander; each
    carries `per_stream` flecks scattered across its width (dense centre, sparse edges), gathered into
    clumps along its length. Fleck length falls from size[0] (px at 1080p, top of region) to size[1]
    (horizon) - perspective - and each fleck is 2-5 overlapping tilted ellipses (dab_angle), eroded with
    fine noise into an irregular flake, with a crisp lit edge toward the sun, a cooler soft underside and
    its own opacity. A faint veil glows under dense clusters. avoid: [(x, y, rx, ry)] thin-out ellipses.
    Colours: lit / body / under (+ haze_col toward the horizon). Straight-alpha RGBA (h, w, 4)."""
    rng = np.random.default_rng(seed)
    sc = w / 1920.0
    y0, y1 = region[0] * h, min(region[1] * h, horizon_y - 2)
    W2, H2 = int(w * ss), int(h * ss)
    m = np.zeros((H2, W2), F32)
    S4 = ss * 4
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    L = math.hypot(w, h) * 1.2
    for si in range(int(streams)):
        # stream anchor spread across the region (perpendicular to the flow)
        t = (si + rng.uniform(0.2, 0.8)) / streams
        yb = y0 + (y1 - y0) * (t if y_bias is None else t ** y_bias)
        xb = w * rng.uniform(*x_range)
        ph = rng.uniform(0, 6.28, 3)
        wid = w * rng.uniform(*width)
        ks = rng.uniform(0.6, 1.4)
        n = int(per_stream * ks * clumps)
        for k in range(n):
            u = rng.uniform(-0.6, 0.6) * L
            # clumps along the stream
            cl = 0.5 + 0.5 * math.sin(u / (0.07 * w) + ph[0]) * math.sin(u / (0.19 * w) + ph[1])
            if rng.random() > 0.05 + 0.95 * cl ** 2:
                continue
            off = rng.normal() * 0.45
            wv = wander * w * math.sin(u / (0.35 * w) + ph[2])
            x = xb + u * ux + (off * wid + wv) * vx
            y = yb + u * uy + (off * wid + wv) * vy
            if not (0 - 0.05 * w < x < w * 1.05 and y0 - 0.05 * h < y < y1):
                continue
            kk = float(np.clip((y - y0) / max(horizon_y - y0, 1), 0, 1))
            pr = float(_ss(y0 - 0.03 * h, y0 + 0.04 * h, y) * (1 - _ss(y1 - 0.1 * h, y1, y)))
            for (ax, ay, rx_, ry_) in avoid:
                d2 = ((x - ax) / rx_) ** 2 + ((y - ay) / ry_) ** 2
                pr *= float(np.clip(d2 - 0.6, 0, 1))
            if rng.random() > pr:
                continue
            L0 = (size[0] + (size[1] - size[0]) * kk ** 0.7) * sc
            centre = math.exp(-off * off * 2.0)
            ln = L0 * (0.25 + 0.95 * rng.random() ** 1.8) * (0.55 + 0.6 * centre)
            op = rng.uniform(0.45, 1.0) * (0.55 + 0.45 * centre)
            da = dab_angle + rng.uniform(-18, 18)
            dar = math.radians(da)
            fx, fy = math.cos(dar), math.sin(dar)
            for q in range(int(rng.integers(2, 6))):
                d = rng.uniform(-0.5, 0.5) * ln
                e = rng.uniform(-0.15, 0.15) * ln
                ex_ = x + d * fx - e * fy
                ey_ = y + d * fy + e * fx
                ra = ln * rng.uniform(0.18, 0.4)
                rb = max(ra * rng.uniform(0.3, 0.6) * (1 - 0.35 * kk), 0.45)
                cv2.ellipse(m, (int(ex_ * S4), int(ey_ * S4)), (max(int(ra * S4 * ss / ss), 1), max(int(rb * S4), 1)),
                            da + rng.uniform(-12, 12), 0, 360, float(op), -1, cv2.LINE_AA, 2)
    # torn edges: erode the smooth coverage with fine, slightly sheared noise
    n1 = _noise(W2, H2, max(W2 / (7.0 * sc * ss), 8), seed + 3, 2, stretch=2.0, angle=-dab_angle)
    n2 = _noise(W2, H2, max(W2 / (2.5 * sc * ss), 16), seed + 4, 1)
    cov = np.clip(m, 0, 1.0)
    cm = _blur(cov, 0.7 * ss)
    M = np.clip((cm - 0.32 + 0.3 * n1 + 0.12 * n2) * 4.0, 0, 1)
    op = np.clip(_blur(m, 1.2 * ss) / (_blur((m > 0).astype(F32), 1.2 * ss) + 1e-4), 0, 1)
    if dry:
        dr = np.clip(0.8 + 0.35 * dry * _noise(W2, H2, max(W2 / (3.0 * sc * ss), 16), seed + 6, 2, stretch=4.0,
                                                  angle=-dab_angle), 0.35, 1.0)
    else:
        dr = 1.0
    A = M * op * dr
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    k = max(1.2 * sc * ss, 1.0)
    rim = np.clip(M - _shift_img(M, -sd[0] * k, -sd[1] * k), 0, 1)          # sunward edge
    und = np.clip(M - _shift_img(M, sd[0] * k * 1.5, sd[1] * k * 1.5), 0, 1)   # away-side edge
    col = np.broadcast_to(np.asarray(body, F32), (H2, W2, 3)).copy()
    tu = np.clip(und * 0.8, 0, 1)[..., None]
    col = col * (1 - tu) + np.asarray(under, F32) * tu
    tl = np.clip(rim * 1.2 + 0.25 * op, 0, 1)[..., None]
    col = col * (1 - tl) + np.asarray(lit, F32) * tl
    # haze toward the horizon
    ys = (np.arange(H2, dtype=F32)[:, None] / ss)
    kk = np.clip((ys - y0) / max(horizon_y - y0, 1), 0, 1)
    hc = np.asarray(haze_col if haze_col is not None else veil_col, F32)
    hz = (kk ** 1.5 * haze_bottom)[..., None]
    col = col * (1 - hz) + hc * hz
    if veil:
        vl = np.clip(_blur(A, 14.0 * sc * ss) * 2.0, 0, 1) * veil
        a2 = A + vl * (1 - A)
        col = (col * A[..., None] + np.asarray(veil_col, F32) * (vl * (1 - A))[..., None]) / np.maximum(a2, 1e-4)[..., None]
        A = a2
    col = _resize(col.astype(F32), w, h, cv2.INTER_AREA)
    A = _resize(A.astype(F32), w, h, cv2.INTER_AREA) * opacity
    out = np.dstack([col, np.clip(A, 0, 1)]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), 3.0)
    return out


def heaps_plate(w, h, clouds, sun_dir=(0.75, -0.65), sun_z=0.95, ramp='noon', seed=0, fringe=0.6, tower=None,
                **paint):
    """Several heaps / towers painted in ONE pass (shared big-form light), later entries nearer.
    clouds = [dict(cx, base_y, height, width[, seed, haze, lean, kind, tone])]. tower: kwargs for tower_heads;
    **paint -> paint_heads. Use for distant heap groups and horizon banks."""
    rng = np.random.default_rng(seed)
    sc = w / 1920.0
    hs = HeadSet()
    tw = dict(tower or {})
    for c in clouds:
        r2 = np.random.default_rng(c.get('seed', int(rng.integers(1 << 30))))
        t_ = dict(tw)
        t_.update(c.get('tower', {}))
        tower_heads(hs, r2, c['cx'], c['base_y'], c['height'], c['width'], lean=c.get('lean', 0.0), sun_dir=sun_dir,
                    kind=c.get('kind', 'heap'), sc=sc, haze=c.get('haze', 0.0), tone=c.get('tone', 0.0), **t_)
    if fringe:
        base = max(c['base_y'] for c in clouds)
        add_fringe(hs, w, h, rng, sun_dir=sun_dir, sc=sc, k=0.15, base_y=base, warm=1.0, aa=0.7 * sc, soft=1.5 * sc,
                   beta=0.1, haze=float(np.mean([c.get('haze', 0.0) for c in clouds])), density=fringe)
    paint.setdefault('base_y', max(c['base_y'] for c in clouds))
    return paint_heads(w, h, hs, sun_dir=sun_dir, sun_z=sun_z, ramp=ramp, seed=seed, **paint)


def utility_pole_plate(w, h, x, top, bottom=None, seed=0, sc=None, color=(0.07, 0.1, 0.19),
                       rim_col=(0.42, 0.52, 0.68), sun_dir=(0.75, -0.65), rim=1.0, ss=2, transformer=True,
                       width=0.016):
    """Foreground Japanese utility pole in near-silhouette (cm5_04 / cm5_02): tapered concrete pole with
    step bolts, two crossarms carrying insulators, a pole-mounted transformer (cylinder + bracket +
    bushings), a streetlight arm, and a thin cool rim light on the edges that face the sun. Returns
    (plate RGBA, anchors) where anchors = list of wire attachment points (plate px), top to bottom."""
    sc = sc if sc is not None else w / 1920.0
    bottom = bottom if bottom is not None else h * 1.1
    W2, H2 = int(w * ss), int(h * ss)
    m = np.zeros((H2, W2), F32)
    hi = np.zeros((H2, W2), F32)
    S = 4 * ss

    def P(px, py):
        return (int(round(px * S)), int(round(py * S)))

    def poly(pts, img=m, v=1.0):
        cv2.fillPoly(img, [np.round(np.asarray(pts) * S).astype(np.int32)], v, cv2.LINE_AA, 2)

    wd = width * w
    poly([[x - wd * 0.55, bottom], [x + wd * 0.55, bottom], [x + wd * 0.36, top], [x - wd * 0.36, top]])
    cv2.ellipse(m, P(x, top), (max(int(wd * 0.36 * S), 1), max(int(wd * 0.12 * S), 1)), 0, 180, 360, 1.0, -1,
                cv2.LINE_AA, 2)
    anchors = []
    th = max(0.006 * h, 1.5 * sc)
    for k, (dy, arm) in enumerate(((0.035, 0.085), (0.1, 0.07))):
        ay = top + dy * h
        aw = arm * w
        poly([[x - aw, ay], [x + aw, ay], [x + aw, ay + th], [x - aw, ay + th]])
        # arm braces
        cv2.line(m, P(x - wd * 0.3, ay + th * 5), P(x - aw * 0.45, ay + th), 1.0, max(int(0.35 * th * S), 1), cv2.LINE_AA, 2)
        cv2.line(m, P(x + wd * 0.3, ay + th * 5), P(x + aw * 0.45, ay + th), 1.0, max(int(0.35 * th * S), 1), cv2.LINE_AA, 2)
        for e in (-0.95, -0.55, 0.55, 0.95):
            ix = x + e * aw
            ih = th * 2.2
            iw = th * 0.9
            poly([[ix - iw * 0.5, ay], [ix + iw * 0.5, ay], [ix + iw * 0.3, ay - ih], [ix - iw * 0.3, ay - ih]])
            for q in range(3):
                cv2.ellipse(m, P(ix, ay - ih * (0.25 + 0.3 * q)), (max(int(iw * 0.8 * S), 1), max(int(th * 0.25 * S), 1)),
                            0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
            anchors.append((ix, ay - ih))
    # step bolts
    for q in range(16):
        by = top + 0.2 * h + q * 0.05 * h
        side = 1 if q % 2 else -1
        cv2.line(m, P(x + side * wd * 0.35, by), P(x + side * wd * 0.62, by - 0.001 * h), 1.0,
                 max(int(th * 0.28 * S), 1), cv2.LINE_AA, 2)
    if transformer:
        ty = top + 0.16 * h
        tw_, th_ = wd * 1.5, 0.075 * h
        tx = x - wd * 0.5 - tw_ * 0.55
        poly([[tx - tw_ * 0.5, ty], [tx + tw_ * 0.5, ty], [tx + tw_ * 0.5, ty + th_], [tx - tw_ * 0.5, ty + th_]])
        cv2.ellipse(m, P(tx, ty), (max(int(tw_ * 0.5 * S), 1), max(int(tw_ * 0.14 * S), 1)), 0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
        cv2.ellipse(m, P(tx, ty + th_), (max(int(tw_ * 0.5 * S), 1), max(int(tw_ * 0.14 * S), 1)), 0, 0, 180, 1.0, -1, cv2.LINE_AA, 2)
        for fy in (0.2, 0.5, 0.8):                             # cooling fins / bands
            cv2.line(m, P(tx - tw_ * 0.56, ty + th_ * fy), P(tx + tw_ * 0.56, ty + th_ * fy), 1.0,
                     max(int(th * 0.35 * S), 1), cv2.LINE_AA, 2)
        poly([[tx + tw_ * 0.4, ty + th_ * 0.3], [x, ty + th_ * 0.25], [x, ty + th_ * 0.35], [tx + tw_ * 0.4, ty + th_ * 0.4]])
        for bx in (-0.25, 0.25):
            poly([[tx + bx * tw_ - th * 0.3, ty - tw_ * 0.1], [tx + bx * tw_ + th * 0.3, ty - tw_ * 0.1],
                  [tx + bx * tw_ + th * 0.2, ty - tw_ * 0.1 - th * 2.2], [tx + bx * tw_ - th * 0.2, ty - tw_ * 0.1 - th * 2.2]])
        # sunlit highlight on the transformer's top rim and right side
        cv2.ellipse(hi, P(tx + tw_ * 0.12, ty - tw_ * 0.02), (max(int(tw_ * 0.34 * S), 1), max(int(tw_ * 0.05 * S), 1)),
                    0, 180, 360, 1.0, -1, cv2.LINE_AA, 2)
        cv2.line(hi, P(tx + tw_ * 0.47, ty + th_ * 0.1), P(tx + tw_ * 0.47, ty + th_ * 0.9), 1.0, max(int(th * 0.3 * S), 1),
                 cv2.LINE_AA, 2)
        anchors.append((tx - tw_ * 0.25, ty - tw_ * 0.1 - th * 2.2))
    # streetlight arm
    ly = top + 0.3 * h
    cv2.line(m, P(x, ly), P(x - 0.05 * w, ly - 0.02 * h), 1.0, max(int(th * 0.7 * S), 1), cv2.LINE_AA, 2)
    cv2.ellipse(m, P(x - 0.052 * w, ly - 0.016 * h), (max(int(0.012 * w * S), 1), max(int(0.004 * h * S), 1)), -18, 0, 360,
                1.0, -1, cv2.LINE_AA, 2)
    M = _resize(np.clip(m, 0, 1), w, h, cv2.INTER_AREA)
    HI = _resize(np.clip(hi, 0, 1), w, h, cv2.INTER_AREA) * M
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    k = 1.6 * sc
    rimm = np.clip(M - _shift_img(M, -sd[0] * k, -sd[1] * k), 0, 1) * rim
    col = np.broadcast_to(np.asarray(color, F32), (h, w, 3)).copy()
    col = col * (1 - rimm[..., None]) + np.asarray(rim_col, F32) * rimm[..., None]
    col = col * (1 - HI[..., None]) + np.asarray((0.85, 0.85, 0.8), F32) * HI[..., None]
    # vertical value gradient: the pole base sinks into shade
    ys = np.arange(h, dtype=F32)[:, None] / h
    col = col * (1 - 0.3 * _ss(0.5, 1.0, ys))[..., None]
    out = np.dstack([col, M]).astype(F32)
    out[..., :3] = _bleed(out[..., :3], (M > 0.02).astype(F32), 2.0)
    anchors.sort(key=lambda p_: (p_[1], p_[0]))
    return out, anchors


def draw_wires(W, H, wires, t=0.0, color=(0.06, 0.08, 0.16), sun=None, glint=1.0, glint_col=(1.0, 0.95, 0.85),
               sway=0.02, sc=None):
    """Per-frame sagging wires (screen px) with gentle sway and a specular glint where they cross the sun
    bloom. wires = [(x0, y0, x1, y1, sag, thickness_px, phase)], sag = mid-span drop (px). Returns an
    RGBA layer (H, W, 4) to composite (colour) + additive glint via [..., :3] premultiplied handling:
    returns (layer, glint_rgb)."""
    sc = sc if sc is not None else W / 1920.0
    m = np.zeros((H, W), F32)
    SH = 4
    u = np.linspace(0, 1, 120)
    for (x0, y0, x1, y1, sag, th, ph) in wires:
        sw = 1.0 + sway * math.sin(0.9 * t + ph) + 0.5 * sway * math.sin(2.1 * t + 1.7 * ph)
        xs = x0 + (x1 - x0) * u
        ys = y0 + (y1 - y0) * u + sag * sw * 4 * u * (1 - u)
        pts = np.round(np.stack([xs, ys], 1) * (1 << SH)).astype(np.int32)
        thi = max(int(round(th)), 1)
        val = min(th, 1.0)
        cv2.polylines(m, [pts], False, float(val), thi, cv2.LINE_AA, SH)
    m = np.clip(m, 0, 1)
    layer = np.dstack([np.broadcast_to(np.asarray(color, F32), (H, W, 3)), m]).astype(F32)
    g = None
    if sun is not None and glint:
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        d = np.sqrt((xx - sun[0]) ** 2 + (yy - sun[1]) ** 2) / W
        g = (m * (np.exp(-d / 0.035) * 1.6 + np.exp(-d / 0.12) * 0.25) * glint)[..., None] * np.asarray(glint_col, F32)
    return layer, g


def iridescence_plate(alpha, center, radius, strength=0.08, band=(0.004, 0.05), seed=0):
    """Faint prismatic fringe in the sky just outside a cloud silhouette near the sun (gow_02 / wwy_04):
    hue cycles with distance from the cloud edge, masked to a soft spot at `center` (px) of `radius`.
    alpha: the cloud plate's alpha (h, w). Returns RGBA (additive-looking, low alpha)."""
    h, w = alpha.shape
    m = (alpha > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5).astype(F32) / w
    t = np.clip((d - band[0]) / (band[1] - band[0]), 0, 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    spot = np.exp(-(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (radius ** 2)))
    ph = t * 2.2 * math.pi
    col = np.stack([0.55 + 0.45 * np.cos(ph), 0.55 + 0.45 * np.cos(ph - 2.1), 0.55 + 0.45 * np.cos(ph - 4.2)], -1)
    a = strength * spot * _ss(0.0, 0.15, t) * (1 - _ss(0.6, 1.0, t)) * (1 - alpha)
    return np.dstack([col.astype(F32), a.astype(F32)])


def deck_plate(w, h, horizon_y, sun, seed=0, fov=55.0, cam_h=1.0, z_near=1.5, z_far=70.0, row_step=0.09,
               head_w=(0.1, 0.5), head_h=0.4, ramp='sunrise', split=None, sun_z=-0.35, rim=(1.5, 0.45),
               rim_px=(1.2, 3.5), rim_focus=0.22, glow=0.6, near_dark=0.3, far_light=0.72, haze=0.85,
               haze_col=None, trough=(0.1, 0.09, 0.24), gap=0.12, clusters=0.6, kuwa=2, value_floor=0.22,
               soft_under=0.5, lift=1.5, sun_col_w=0.35, crust=0.45, extra_rows=None, paint=None,
               floor_rows=1):
    """Sea of clouds seen from above (yn_02 / wwy_02), painted with the heads engine.

    Rows of cauliflower heads laid on a deck plane in perspective (cam at height cam_h looking at the
    horizon): distance z runs from z_far to z_near in log steps (row_step), head widths are drawn in world
    units (head_w, power law) so heads are tiny and dense at the horizon and large near the camera. Heads
    are painted far -> near; every head is lit from the sun's screen position (per-head light vector,
    sun_z < 0 = backlit): crust-lit crowns + a hot rim on the sun-facing crest (rim = (near sun, far from
    sun) strength, rim_px = (far, near) width at 1080p) gathered toward the sun column (rim_focus = width
    as a fraction of w) and toward the horizon; bodies fall into clean cool violet, darker toward the viewer
    (near_dark); aerial haze toward the horizon (haze); dark troughs between rows (trough colour, gap =
    probability of a wider break). clusters: heads gathered into larger merged masses. split=(d1, d2, ...)
    -> [far, ..., nearest] plates for parallax, each opaque below its first row (no holes when shifted)."""
    rng = np.random.default_rng(seed)
    R = RAMPS[ramp] if isinstance(ramp, str) else ramp
    sc = w / 1920.0
    f = (w / 2.0) / math.tan(math.radians(fov) / 2)
    sx, sy = sun
    zs = []
    z = z_near
    while z < z_far:
        zs.append(z)
        z *= 1.0 + row_step * rng.uniform(0.75, 1.25)
    zs = zs[::-1]
    bounds = sorted(split) if split is not None else []
    nb_ = len(bounds) + 1
    sets = [HeadSet() for _ in range(nb_)]
    first_y = [None] * nb_
    rows_y = [[] for _ in range(nb_)]
    for z in zs:
        bi = nb_ - 1 - int(np.searchsorted(np.asarray(bounds, np.float64), z, side='right'))
        ppu = f / z
        yb = horizon_y + f * cam_h / z
        kz = float(np.clip(math.log(z / z_near) / math.log(z_far / z_near), 0, 1))    # 0 near .. 1 far
        if first_y[bi] is None or yb < first_y[bi]:
            first_y[bi] = yb
        rows_y[bi].append(yb)
        Xmax = (w * 0.62) / f * z
        X = -Xmax - rng.uniform(0, 1.0)
        items = []
        while X < Xmax:
            wd = head_w[0] + (head_w[1] - head_w[0]) * rng.random() ** 1.8
            if rng.random() < clusters * 0.35:
                wd *= rng.uniform(1.6, 2.6)                    # a merged larger mass
            rx = wd * 0.5 * ppu
            ry = rx * head_h * rng.uniform(0.7, 1.2) * (0.6 + 0.4 * (1 - kz))
            dz = rng.uniform(-0.3, 0.3) * z * row_step
            ppu2 = f / (z + dz)
            x = w / 2.0 + X * ppu2
            y = horizon_y + f * cam_h / (z + dz) - ry * rng.uniform(0.35, 0.7) * (1 - 0.4 * kz)
            items.append((rng.random(), x, y, rx, ry, dz))
            X += wd * rng.uniform(0.45, 0.8)
        # break the row with an occasional wide gap (dark trough shows)
        if rng.random() < gap:
            gx = rng.uniform(0.05, 0.95) * w
            gw = rng.uniform(1.5, 4.0) * head_w[1] * ppu
            items = [it for it in items if abs(it[1] - gx) > gw]
        items.sort(key=lambda t: t[0])
        for (_, x, y, rx, ry, dz) in items:
            if x + rx < -0.05 * w or x - rx > 1.05 * w or rx < 0.6 * sc:
                continue
            # light: toward the sun's screen position, slightly lifted (painter's cheat: crowns catch it)
            lx_, ly_ = sx - x, sy - y - lift * abs(sx - x) * 0.2
            n_ = math.hypot(lx_, ly_) + 1e-9
            L = np.array([lx_ / n_, ly_ / n_, sun_z])
            L = L / np.linalg.norm(L)
            foc = math.exp(-((x - sx) / (rim_focus * w)) ** 2)
            near = 1 - kz
            ra = (rim[1] + (rim[0] - rim[1]) * foc) * (0.35 + 0.65 * kz ** 0.5 + 0.3 * foc)
            rp = (rim_px[0] + (rim_px[1] - rim_px[0]) * near ** 1.5) * sc
            hz = haze * kz ** 1.6
            gl = glow * foc * (0.4 + 0.6 * kz)
            tone = -near_dark * near ** 1.3 * (1 - 0.5 * foc) + far_light * kz ** 2 * 0.3
            mp = max(0.8 * sc, 0.35 * sc + rx * 0.02)
            sets[bi].add(x, y, rx, ry, rng, k=0.75, haze=hz, tone=tone, aa=0.6 * sc,
                         soft=max(soft_under * ry, 1.2 * sc), rim=ra, rim_px=rp, glow=gl, warm=1.0, beta=0.0,
                         bump=(0.1, 0.3), arc=(-200.0, 20.0), lumps=(1, 3) if rx > 12 * sc else (0, 1),
                         lump_r=(0.3, 0.55), sub=(0.25 if rx > 90 * sc else 0.6) if rx > 20 * sc else 0.2,
                         sub3=0.2 if rx > 90 * sc else 0.35, min_px=mp,
                         light=tuple(L), sun_ang=math.atan2(ly_, lx_))
    out = []
    for bi in range(nb_):
        hs = sets[bi]
        if len(hs) == 0:
            out.append(np.zeros((h, w, 4), F32))
            continue
        # value field: brighter toward the sun column and the horizon, darker toward the viewer
        ys = np.arange(h, dtype=F32)[:, None]
        xs = np.arange(w, dtype=F32)[None, :]
        zz = f * cam_h / np.maximum(ys - horizon_y, 1.0)
        kz = np.clip(np.log(np.maximum(zz, z_near) / z_near) / math.log(z_far / z_near), 0, 1)
        col = np.exp(-((xs - sx) / (sun_col_w * w)) ** 2)
        Vb = (value_floor + 0.4 * kz ** 1.0 + 0.3 * col * (0.3 + 0.7 * kz)).astype(F32)
        Vb = np.broadcast_to(Vb, (h, w)).astype(F32).copy()
        pk = dict(crust=crust, crust_mix=0.75, poster=0.4, brush=0.04, kuwa=kuwa, seed=seed + bi, k_inside=0.9,
                  detail_depth=40.0 * sc, soft_inside=0.25, edge_noise=1.0, warm_split=0.9, haze_col=haze_col,
                  crop=False, close_px=3.0, rim_sil=False, rim_replace=0.9, rim_up=0.85, crust_bthr=0.6)
        pk.update(paint or {})
        pl = paint_heads(w, h, hs, sun_dir=(0.0, -1.0), sun_z=sun_z, ramp=R, Vbig=Vb, **pk)
        # opaque trough floor under the band's first row (dark, hazier far away): no holes under parallax
        if first_y[bi] is not None:
            ry_ = sorted(rows_y[bi])
            y0 = ry_[min(floor_rows, len(ry_) - 1)] if bi < nb_ - 1 or True else first_y[bi]
            ysf = np.arange(h, dtype=F32)[:, None]
            zz = f * cam_h / np.maximum(ysf - horizon_y, 1.0)
            kzf = np.clip(np.log(np.maximum(zz, z_near) / z_near) / math.log(z_far / z_near), 0, 1)
            hc = np.asarray(haze_col if haze_col is not None else R['haze'], F32)
            fl = np.asarray(trough, F32) * (1 - haze * kzf ** 1.6)[..., None] * 1.0 + hc * (haze * kzf ** 1.6)[..., None]
            fa = _ss(y0 - 2 * sc, y0 + 6 * sc, ysf) * np.ones((1, w), F32)
            fl = np.broadcast_to(fl, (h, w, 3))
            A = pl[..., 3]
            a2 = A + fa * (1 - A)
            rgb = (pl[..., :3] * A[..., None] + fl * (fa * (1 - A))[..., None]) / np.maximum(a2, 1e-4)[..., None]
            pl = np.dstack([rgb, a2]).astype(F32)
            pl[..., :3] = _bleed(pl[..., :3], (pl[..., 3] > 0.02).astype(F32), 3.0)
        out.append(pl)
    return out if split is not None else out[0]


# =========================================================================================================
# VOLUMETRIC PAINTER (final) - ray-marched lobe volumes / heightfield deck / sky-plane cloudlets, painted
# =========================================================================================================
# See the module docstring (FINAL API). Everything below is procedural: numba ray marching of analytic
# lobe clusters (voxelised SDF + billow erosion), a sun march for light, then a painterly pass (value ramp,
# Kuwahara flattening, warm/cool drift, dry-brush strokes, crisp-lit / lost-shadow edges, silhouette florets).
import sys as _sys
K = _sys.modules[__name__]

# ----------------------------------------------------------------------------- noise
@njit(cache=True, inline='always')
def _h3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = h & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / 16777215.0


@njit(cache=True)
def vnoise3(x, y, z, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    uz = fz * fz * (3 - 2 * fz)
    ix = int(ix)
    iy = int(iy)
    iz = int(iz)
    a = _h3(ix, iy, iz, seed)
    b = _h3(ix + 1, iy, iz, seed)
    c = _h3(ix, iy + 1, iz, seed)
    d = _h3(ix + 1, iy + 1, iz, seed)
    e = _h3(ix, iy, iz + 1, seed)
    f = _h3(ix + 1, iy, iz + 1, seed)
    g = _h3(ix, iy + 1, iz + 1, seed)
    h = _h3(ix + 1, iy + 1, iz + 1, seed)
    k0 = a + (b - a) * ux
    k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux
    k3 = g + (h - g) * ux
    l0 = k0 + (k1 - k0) * uy
    l1 = k2 + (k3 - k2) * uy
    return (l0 + (l1 - l0) * uz) * 2.0 - 1.0


@njit(cache=True)
def billow3(x, y, z, seed, octaves):
    """Billowy fbm (sum of |noise|, inverted -> round puffs) in about -1..1."""
    s = 0.0
    a = 0.5
    nrm = 0.0
    x, y = 0.83 * x + 0.56 * y, -0.56 * x + 0.83 * y
    for o in range(octaves):
        n = vnoise3(x, y, z, seed + o * 17)
        s += a * (1.0 - 2.0 * abs(n))
        nrm += a
        # rotate between octaves (kills lattice-aligned creases)
        x2 = (0.00 * x + 0.80 * y + 0.60 * z) * 2.03 + 1.7
        y2 = (-0.80 * x + 0.36 * y - 0.48 * z) * 2.03 - 3.1
        z2 = (-0.60 * x - 0.48 * y + 0.64 * z) * 2.03 + 0.9
        x, y, z = x2, y2, z2
        a *= 0.5
    return s / nrm


# ----------------------------------------------------------------------------- sdf grid
@njit(parallel=True, cache=True)
def _splat(E, ox, oy, oz, vs, nx, ny, nz, far):
    """E (N, 13) lobes: c(3) r(3) cut_hi cut_lo ... -> sdf-ish grid (nz, ny, nx), world units."""
    D = np.full((nz, ny, nx), far, np.float32)
    N = E.shape[0]
    for iz in prange(nz):
        pz = oz + (iz + 0.5) * vs
        for i in range(N):
            cz, rz = E[i, 2], E[i, 5]
            m = far
            if abs(pz - cz) > rz + m:
                continue
            cx, cy, rx, ry = E[i, 0], E[i, 1], E[i, 3], E[i, 4]
            ch, cl = E[i, 6], E[i, 7]
            x0 = max(int((cx - rx - m - ox) / vs), 0)
            x1 = min(int((cx + rx + m - ox) / vs) + 1, nx)
            y0 = max(int((cy - ry - m - oy) / vs), 0)
            y1 = min(int((cy + ry + m - oy) / vs) + 1, ny)
            rmin = min(rx, min(ry, rz))
            for iy in range(y0, y1):
                py = oy + (iy + 0.5) * vs
                for ix in range(x0, x1):
                    px = ox + (ix + 0.5) * vs
                    qx = (px - cx) / rx
                    qy = (py - cy) / ry
                    qz = (pz - cz) / rz
                    k0 = math.sqrt(qx * qx + qy * qy + qz * qz)
                    k1 = math.sqrt((qx / rx) ** 2 + (qy / ry) ** 2 + (qz / rz) ** 2) + 1e-9
                    d = k0 * (k0 - 1.0) / k1
                    if k0 < 1e-6:
                        d = -rmin
                    # flat cuts (base plane / top plane)
                    d = max(d, py - cl)
                    d = max(d, ch - py)
                    if d < D[iz, iy, ix]:
                        D[iz, iy, ix] = d
    return D


@njit(cache=True, inline='always')
def _tri(G, fx, fy, fz):
    nz, ny, nx = G.shape
    fx = min(max(fx - 0.5, 0.0), nx - 1.001)
    fy = min(max(fy - 0.5, 0.0), ny - 1.001)
    fz = min(max(fz - 0.5, 0.0), nz - 1.001)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    ux = fx - ix
    uy = fy - iy
    uz = fz - iz
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    iz1 = min(iz + 1, nz - 1)
    a = G[iz, iy, ix] + (G[iz, iy, ix1] - G[iz, iy, ix]) * ux
    b = G[iz, iy1, ix] + (G[iz, iy1, ix1] - G[iz, iy1, ix]) * ux
    c = G[iz1, iy, ix] + (G[iz1, iy, ix1] - G[iz1, iy, ix]) * ux
    d = G[iz1, iy1, ix] + (G[iz1, iy1, ix1] - G[iz1, iy1, ix]) * ux
    e = a + (b - a) * uy
    f = c + (d - c) * uy
    return e + (f - e) * uz


@njit(parallel=True, cache=True)
def _erode(D, ox, oy, oz, vs, amp, freq, seed, octaves, top_boost, ytop, ybase):
    """Add billow noise to the sdf near the surface (cauliflower texture), stronger toward the top."""
    nz, ny, nx = D.shape
    for iz in prange(nz):
        pz = oz + (iz + 0.5) * vs
        for iy in range(ny):
            py = oy + (iy + 0.5) * vs
            s = min(max((ybase - py) / max(ybase - ytop, 1e-6), 0.0), 1.0)
            a = amp * (1.0 + top_boost * s)
            for ix in range(nx):
                d = D[iz, iy, ix]
                if d > 3.0 * a + 2 * vs or d < -3.0 * a - 2 * vs:
                    continue
                px = ox + (ix + 0.5) * vs
                n = billow3(px * freq, py * freq, pz * freq, seed, octaves)
                D[iz, iy, ix] = d - a * (n + 0.2)


@njit(parallel=True, cache=True)
def _density(D, soft):
    nz, ny, nx = D.shape
    out = np.zeros_like(D)
    for iz in prange(nz):
        for iy in range(ny):
            for ix in range(nx):
                v = 0.5 - D[iz, iy, ix] / (2.0 * soft)
                out[iz, iy, ix] = min(max(v, 0.0), 1.0)
    return out


@njit(parallel=True, cache=True)
def _march_od(Dn, lx, ly, lz, step, maxd):
    """Optical depth (in voxel units x density) from each voxel toward direction L (grid space)."""
    nz, ny, nx = Dn.shape
    out = np.zeros_like(Dn)
    for iz in prange(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = ix + 0.5
                y = iy + 0.5
                z = iz + 0.5
                od = 0.0
                t = step * 0.5
                while t < maxd:
                    px = x + lx * t
                    py = y + ly * t
                    pz = z + lz * t
                    if px < 0 or py < 0 or pz < 0 or px > nx or py > ny or pz > nz:
                        break
                    od += _tri(Dn, px, py, pz) * step
                    t += step
                out[iz, iy, ix] = od
    return out


def _down2(G):
    nz, ny, nx = G.shape
    g = G[:nz // 2 * 2, :ny // 2 * 2, :nx // 2 * 2]
    return (g[0::2, 0::2, 0::2] + g[1::2, 0::2, 0::2] + g[0::2, 1::2, 0::2] + g[0::2, 0::2, 1::2] +
            g[1::2, 1::2, 0::2] + g[1::2, 0::2, 1::2] + g[0::2, 1::2, 1::2] + g[1::2, 1::2, 1::2]) / 8.0


@njit(parallel=True, cache=True)
def _render(out, x0, y0, f, ppx, ppy, D, soft_g, ODs, ODa, ox, oy, oz, vs, sd2, sd4, sig, sig_a,
            fine_amp, fine_freq, seed, Lx, Ly, Lz, g1, g2, gmix, ms_a, ms_b, n_ms, step_k, sig_l, NX, NY, NZ):
    """Ray march. out (h, w, 6): sun radiance, ambient, alpha, depth(avg), sun-od(avg), height."""
    hh, ww = out.shape[0], out.shape[1]
    nz, ny, nx = D.shape
    gx1, gy1, gz1 = ox + nx * vs, oy + ny * vs, oz + nz * vs
    for j in prange(hh):
        for i in range(ww):
            dx = (x0 + i + 0.5) - ppx
            dy = (y0 + j + 0.5) - ppy
            dz = f
            dn = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dn
            dy /= dn
            dz /= dn
            # slab intersect
            tn = 0.0
            tf = 1e30
            if abs(dx) > 1e-9:
                t1 = ox / dx
                t2 = gx1 / dx
                tn = max(tn, min(t1, t2))
                tf = min(tf, max(t1, t2))
            if abs(dy) > 1e-9:
                t1 = oy / dy
                t2 = gy1 / dy
                tn = max(tn, min(t1, t2))
                tf = min(tf, max(t1, t2))
            t1 = oz / dz
            t2 = gz1 / dz
            tn = max(tn, min(t1, t2))
            tf = min(tf, max(t1, t2))
            if tf <= tn:
                continue
            cosT = dx * Lx + dy * Ly + dz * Lz
            # dual HG phase
            ph1 = (1 - g1 * g1) / (4 * math.pi * (1 + g1 * g1 - 2 * g1 * cosT) ** 1.5)
            ph2 = (1 - g2 * g2) / (4 * math.pi * (1 + g2 * g2 - 2 * g2 * cosT) ** 1.5)
            ph = (ph1 * (1 - gmix) + ph2 * gmix) * 4 * math.pi
            T = 1.0
            Ssun = 0.0
            Samb = 0.0
            Wd = 0.0
            Wod = 0.0
            Wy = 0.0
            Wl = 0.0
            base_step = step_k * vs
            t = tn + base_step * _h3(i, j, 7, seed)
            while t < tf and T > 0.004:
                px = dx * t
                py = dy * t
                pz = dz * t
                fx = (px - ox) / vs
                fy = (py - oy) / vs
                fz = (pz - oz) / vs
                d = _tri(D, fx, fy, fz)
                if d > 3.0 * vs + fine_amp * 2:
                    t += max(d - 2.0 * vs - fine_amp * 2, base_step)
                    continue
                sf = _tri(soft_g, fx, fy, fz)
                if fine_amp > 0:
                    n = billow3(px * fine_freq, py * fine_freq, pz * fine_freq, seed + 99, 2)
                    d = d - fine_amp * (n + 0.2)
                rho = min(max((0.6 * vs - d) / sf, 0.0), 1.0)
                if rho > 0.0:
                    ods = _tri(ODs, fx / 2.0, fy / 2.0, fz / 2.0) * sd2
                    oda = _tri(ODa, fx / 4.0, fy / 4.0, fz / 4.0) * sd4
                    # multiple scattering octaves (Wrenninge)
                    ls = 0.0
                    a_ = 1.0
                    b_ = 1.0
                    for k in range(n_ms):
                        ls += b_ * math.exp(-sig_l * ods * a_) * (ph * (0.6 ** k) + (1 - 0.6 ** k))
                        a_ *= ms_a
                        b_ *= ms_b
                    la = math.exp(-sig_a * oda)
                    ext = sig * rho
                    dt = base_step
                    tr = math.exp(-ext * dt)
                    w_ = T * (1.0 - tr)
                    Ssun += w_ * ls
                    Samb += w_ * la
                    Wd += w_ * pz
                    Wod += w_ * ods
                    Wy += w_ * py
                    nx_ = _tri(NX, fx / 2.0, fy / 2.0, fz / 2.0)
                    ny_ = _tri(NY, fx / 2.0, fy / 2.0, fz / 2.0)
                    nz_ = _tri(NZ, fx / 2.0, fy / 2.0, fz / 2.0)
                    nn_ = math.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) + 1e-6
                    Wl += w_ * (nx_ * Lx + ny_ * Ly + nz_ * Lz) / nn_
                    T *= tr
                t += base_step
            A = 1.0 - T
            out[j, i, 0] = Ssun
            out[j, i, 1] = Samb
            out[j, i, 2] = A
            if A > 1e-4:
                out[j, i, 3] = Wd / A
                out[j, i, 4] = Wod / A
                out[j, i, 5] = Wy / A
                out[j, i, 6] = Wl / A


def build_volume(E, max_dim=300, pad=0.06, erode=None, soft=(1.0, 3.0), seed=0, L=(0.6, -0.7, -0.3),
                 amb_dirs=None, sig=None):
    """E: lobe array (N, 13) in camera space. Returns dict of grids."""
    E = np.ascontiguousarray(E, np.float64)
    lo = np.array([(E[:, 0] - E[:, 3]).min(), (E[:, 1] - E[:, 4]).min(), (E[:, 2] - E[:, 5]).min()])
    hi = np.array([(E[:, 0] + E[:, 3]).max(), (E[:, 1] + E[:, 4]).max(), (E[:, 2] + E[:, 5]).max()])
    ext = hi - lo
    lo = lo - ext * pad
    hi = hi + ext * pad
    ext = hi - lo
    vs = float(ext.max() / max_dim)
    nx, ny, nz = [int(math.ceil(e / vs)) for e in ext]
    amp_max = (erode[0] * (1 + erode[3]) * 1.3 if erode is not None else 0.0) * vs
    D = _splat(E, lo[0], lo[1], lo[2], vs, nx, ny, nz, 2.5 * vs + amp_max)
    if erode is not None:
        amp, wl, octv, top_boost = erode
        ytop = float((E[:, 1] - E[:, 4]).min())
        ybase = float((E[:, 1] + E[:, 4]).max())
        _erode(D, lo[0], lo[1], lo[2], vs, amp * vs, 1.0 / (wl * vs), seed, octv, top_boost, ytop, ybase)
    return dict(D=D, o=lo, vs=vs, n=(nx, ny, nz))


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def vol_render(E, w, h, f, ppx, ppy, L, max_dim=300, erode=(2.5, 7.0, 3, 1.0), soft_lit=1.2, soft_shade=3.0,
               soft_down=3.0, sig=None, sig_a=None, fine_amp=0.8, fine_wl=2.2, seed=0, g=(0.55, -0.2), gmix=0.3,
               ms=(0.35, 0.55, 4), amb_dirs=((0.1, -1, -0.15), (0.2, -0.6, -0.75), (0.75, -0.6, 0.1), (-0.7, -0.65, -0.2), (-0.3, -0.3, -0.9)),
               crop=True, step_k=0.4, sig_l=None, od_blur=1.0, form=0.55, form_blur=5.0):
    """Volumetric render of a lobe cluster. Returns fields dict (h, w): sun, amb, A, depth, od, y + bbox."""
    L = _unit(L)
    vol = build_volume(E, max_dim=max_dim, erode=erode, seed=seed)
    D, o, vs = vol['D'], vol['o'], vol['vs']
    sig = sig if sig is not None else 0.9 / vs
    sig_a = sig_a if sig_a is not None else 0.12 / vs
    sig_l = sig_l if sig_l is not None else 0.3 / vs
    Dn0 = _density(D, soft_lit * vs)
    D2 = np.ascontiguousarray(_down2(Dn0)).astype(F32)
    ODs2 = _march_od(D2, L[0], L[1], L[2], 0.7, 1e9)
    ODs2 = _gblur3(ODs2, od_blur)
    # softness grid: lit -> crisp, sun-shadowed or down-facing -> lost
    od_w = ODs2 * 2 * vs
    tr = np.exp(-sig * od_w * 0.35)
    tr_f = _up2(tr, D.shape)
    gyy = np.gradient(D, axis=1) / vs
    soft = vs * (soft_lit + (soft_shade - soft_lit) * (1 - tr_f) + soft_down * np.clip(gyy, 0, 1))
    soft = _gblur3(np.ascontiguousarray(soft, dtype=F32), 1.5)
    Dn = _density(D, soft_lit * vs * 1.5)
    D4 = np.ascontiguousarray(_down2(_down2(Dn))).astype(F32)
    oda = np.zeros_like(D4)
    for d_ in amb_dirs:
        u = _unit(d_)
        oda += _march_od(D4, u[0], u[1], u[2], 0.7, 1e9)
    oda /= len(amb_dirs)
    oda = _gblur3(oda, 1.0)
    E_ = np.asarray(E)
    # screen bbox
    if crop:
        nx_, ny_, nz_ = vol['n']
        cs = np.array([[o[0] + i * nx_ * vs, o[1] + j * ny_ * vs, o[2] + k * nz_ * vs]
                       for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        sx = f * cs[:, 0] / np.maximum(cs[:, 2], 1e-3) + ppx
        sy = f * cs[:, 1] / np.maximum(cs[:, 2], 1e-3) + ppy
        x0, x1 = int(max(sx.min(), 0)), int(min(sx.max() + 1, w))
        y0, y1 = int(max(sy.min(), 0)), int(min(sy.max() + 1, h))
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    out = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1), 7), F32)
    # surface normals (half res): a blend of the local lobe normal and a heavily blurred big-form normal
    Dh = np.ascontiguousarray(_down2(D)).astype(F32)
    Nf = np.gradient(_gblur3(Dh, 0.8))
    Nb = np.gradient(_gblur3(Dh, form_blur))
    def _nz(g):
        return g / (np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-6)
    Nf = _nz(np.array(Nf))
    Nb = _nz(np.array(Nb))
    Nm = Nf * (1 - form) + Nb * form
    NZg, NYg, NXg = [np.ascontiguousarray(Nm[k]).astype(F32) for k in range(3)]
    _render(out, float(x0), float(y0), float(f), float(ppx), float(ppy), D, soft, np.ascontiguousarray(ODs2),
            np.ascontiguousarray(oda.astype(F32)), o[0], o[1], o[2], vs, 2 * vs, 4 * vs, sig, sig_a,
            fine_amp * vs, 1.0 / (fine_wl * vs), seed, L[0], L[1], L[2], g[0], g[1], gmix, ms[0], ms[1], int(ms[2]), step_k, sig_l, NXg, NYg, NZg)
    return dict(sun=out[..., 0], amb=out[..., 1], A=out[..., 2], depth=out[..., 3], od=out[..., 4], y=out[..., 5],
                lam=out[..., 6], box=(x0, y0, x1, y1), vs=vs)


def _up2(G, shape):
    """Trilinear upsample of a half-res grid to shape."""
    from scipy.ndimage import zoom
    z = zoom(G, (shape[0] / G.shape[0], shape[1] / G.shape[1], shape[2] / G.shape[2]), order=1, mode='nearest')
    out = np.empty(shape, F32)
    out[...] = z[:shape[0], :shape[1], :shape[2]] if z.shape == tuple(shape) else np.pad(
        z, [(0, max(shape[i] - z.shape[i], 0)) for i in range(3)], mode='edge')[:shape[0], :shape[1], :shape[2]]
    return out


def tower_lobes(rng, X, Yb, Z, width, height, lean=0.1, sun3=None, n_main=9, taper=0.45, depth=0.8,
                sizes=(0.2, 0.1, 0.05), dens=0.9, front=0.85, hier=0.8, clump=0.6, crown=0.3, base_cut=None,
                side_towers=2, recede=0.25):
    """Towering cumulus as stacked big masses (tower-on-tower, sizes shrinking upward, leaning), medium
    heads grown on the upper / sunward surface and fine cauliflower mostly on the silhouette. Camera
    space, px units at depth. Returns lobe array (N, 13)."""
    hw = 0.5 * width
    cores = []
    ph = rng.uniform(0, 6.28, 2)

    def ax(s):
        return X + lean * height * s + 0.12 * hw * math.sin(2.3 * math.pi * s + ph[0])

    s = 0.0
    k = 0
    while s < 1.0 and k < 40:
        pw = hw * (1.0 - taper * s)
        r = pw * rng.uniform(0.55, 0.75)
        if s + r / height * 0.9 > 1.0:
            r = max((1.0 - s) * height / 0.9, 0.25 * pw)
        y = Yb - s * height - r * 0.55
        n_side = 1 if s > 0.75 else 2
        for m in range(n_side):
            off = (m - (n_side - 1) / 2) * pw * rng.uniform(0.6, 0.9) + rng.uniform(-0.15, 0.15) * pw
            rr = r * rng.uniform(0.85, 1.1) / (1.0 if n_side == 1 else 1.15)
            z = Z + rng.uniform(-0.2, 0.2) * hw * depth + recede * s * hw
            cores.append((np.array([ax(s) + off, y + rng.uniform(-0.1, 0.1) * rr, z]),
                          np.array([rr * rng.uniform(1.0, 1.15), rr * rng.uniform(0.8, 0.95), rr * depth * 1.1])))
        s += r / height * rng.uniform(0.75, 1.0)
        k += 1
    # side towers: lower turrets leaning on the main column (stepped massing)
    for t in range(side_towers):
        side = 1 if (t % 2 == 0) == (rng.random() < 0.5) else -1
        th = height * rng.uniform(0.3, 0.5)
        tw = hw * rng.uniform(0.5, 0.7)
        xb = X + side * hw * rng.uniform(0.75, 1.0)
        zb = Z + rng.uniform(-0.4, 0.2) * hw
        s = 0.0
        while s < 1.0:
            r = tw * (1 - 0.35 * s) * rng.uniform(0.6, 0.8)
            y = Yb - s * th - r * 0.5
            cores.append((np.array([xb + side * 0.15 * th * s, y, zb]),
                          np.array([r * 1.1, r * 0.85, r * depth * 1.1])))
            s += r / th * 0.9
    lob = K.surface_grow(rng, cores, [hw * f for f in sizes], dens=dens, area=math.pi * hw * height * 1.3,
                         sun=sun3, flat=0.85, clump=clump, front=front, hier=hier)
    cut = Yb if base_cut is None else base_cut
    rows = []
    for c, r, rt in lob:
        rows.append((c[0], c[1], c[2], r[0], r[1], r[2], -1e30, cut + rng.normal() * 0.004 * height,
                     1.0, 0.0, 0.0, rt, 0.0))
    return np.asarray(rows, np.float64)


def _gblur3(G, s):
    if s <= 0:
        return G
    from scipy.ndimage import gaussian_filter
    return np.ascontiguousarray(gaussian_filter(G, s, mode='nearest')).astype(F32)


VRAMPS = {
    'noon': dict(stops=[(0.0, (0.38, 0.52, 0.78)), (0.22, (0.53, 0.67, 0.88)), (0.42, (0.71, 0.8, 0.93)),
                        (0.6, (0.87, 0.9, 0.95)), (0.8, (0.99, 0.97, 0.92)), (1.0, (1.04, 1.0, 0.91))],
                 hi=(1.06, 1.02, 0.94)),
    'sunrise': dict(stops=[(0.0, (0.11, 0.1, 0.27)), (0.2, (0.2, 0.17, 0.4)), (0.4, (0.36, 0.28, 0.54)),
                           (0.58, (0.62, 0.42, 0.6)), (0.74, (0.95, 0.62, 0.55)), (0.88, (1.15, 0.84, 0.6)),
                           (1.0, (1.3, 1.05, 0.78))],
                    hi=(1.3, 1.05, 0.8)),
}


def paint_vol(R, w, h, ramp='noon', k_sun=0.85, k_amb=0.3, gamma=0.8, crisp=(0.12, 0.55), lost=(0.0, 1.0),
              kuwa=4, brush=0.03, seed=0, steps=0.35, step_pts=(0.3, 0.55, 0.78), step_soft=0.05, base_y=None,
              base_dark=0.0, base_h=0.1, frill=0.0, frill_px=3.0, frill_lit=True, tone_k=0.5, lam_floor=0.35, lam_wrap=0.3,
              tint_var=0.5, strokes=0.025, stroke_angle=-35.0, backlit=None):
    """Paint volumetric fields into a straight-alpha RGBA plate (h, w, 4)."""
    x0, y0, x1, y1 = R['box']
    A = R['A']
    ia = 1.0 / np.maximum(A, 1e-4)
    S = R['sun'] * ia
    Am = R['amb'] * ia
    sc = w / 1920.0
    lam = np.clip((R['lam'] + lam_wrap) / (1 + lam_wrap), 0, 1)
    S = S * (lam_floor + (1 - lam_floor) * lam)
    v = np.clip(k_sun * S + k_amb * Am, 0, None)
    v = (1.0 - np.exp(-v * tone_k)) / (1.0 - math.exp(-tone_k))      # highlight roll-off keeps lit-side form
    v = np.clip(v, 0, None) ** gamma
    hh, ww = A.shape
    if brush:
        v = v + brush * K._noise(ww, hh, max(ww / (26.0 * sc), 4), seed + 3, 3, stretch=2.5, angle=-30)
    if steps:
        wn = K._noise(ww, hh, max(ww / (70.0 * sc), 3), seed + 4, 3) * 0.05
        q = np.zeros_like(v)
        for t_ in step_pts:
            q += K._ss(t_ - step_soft, t_ + step_soft, v + wn)
        q = q / len(step_pts)
        v = v * (1 - steps) + (0.08 + 0.94 * q + 0.1 * (v - 0.5)) * steps
    if base_y is not None and base_dark:
        ys = np.arange(y0, y1, dtype=F32)[:, None]
        v = v * (1 - base_dark * (1 - K._ss(0, base_h * h, base_y - ys)))
    if backlit is not None:
        # backlit (sun behind the cloud): cool body from the sky light, hot rim / translucency from the sun term
        bl = dict(body=((0.0, (0.1, 0.09, 0.24)), (0.5, (0.24, 0.2, 0.44)), (1.0, (0.46, 0.38, 0.62))),
                  rim=(1.45, 0.92, 0.55), k_rim=0.5, hot=(1.6, 1.25, 0.9), top_warm=(0.5, 0.3, 0.3))
        bl.update(backlit)
        am = np.clip(Am, 0, 1)
        col = _ramp(bl['body'], am)
        sr = np.clip(S * bl['k_rim'], 0, 3)
        col = col + np.asarray(bl['rim'], F32) * np.minimum(sr, 1.0)[..., None] +             np.asarray(bl['hot'], F32) * np.clip(sr - 1.0, 0, 1)[..., None] * 0.5
    else:
        col = _ramp(VRAMPS[ramp]['stops'], np.clip(v, 0, 1))
    if kuwa:
        col = K.kuwahara(col, max(1, int(round(kuwa * sc))), q=6.0)
    if tint_var:
        # painted colour variation: slow warm / cool drift across the masses (no flat vector fills)
        tn = K._noise(ww, hh, max(ww / (160.0 * sc), 2), seed + 31, 3)
        warm = np.array([1.0, 0.985, 0.95], F32)
        cool = np.array([0.95, 0.97, 1.02], F32)
        k_ = np.clip(tn * 0.5 + 0.5, 0, 1)[..., None]
        col = col * (1 + tint_var * ((warm - 1) * k_ + (cool - 1) * (1 - k_)) * 2)
    if strokes:
        # visible dry-brush strokes (after the flattening, so they survive)
        st = K._noise(ww, hh, max(ww / (9.0 * sc), 6), seed + 41, 2, stretch=5.0, angle=stroke_angle)
        col = col * (1 + strokes * st[..., None])
    # edges: crisp where lit, lost where shaded
    lit = K._ss(0.25, 0.7, S)
    a0 = crisp[0] * lit + lost[0] * (1 - lit)
    a1 = crisp[1] * lit + lost[1] * (1 - lit)
    Ae = np.clip((A - a0) / np.maximum(a1 - a0, 1e-3), 0, 1)
    Ae = Ae * Ae * (3 - 2 * Ae)
    if frill:
        # dry-brush cauliflower grains on the silhouette: widen the edge, re-threshold with fine billow noise
        fs = frill_px * sc
        Ab = cv2.GaussianBlur(Ae, (0, 0), max(fs * 1.3, 0.6))
        n1 = _billow2(ww, hh, fs * 2.2, seed + 17)
        n2 = _billow2(ww, hh, fs * 0.9, seed + 18)
        nn = 0.65 * n1 + 0.35 * n2
        up = lit if frill_lit else 1.0
        th = 0.5 + frill * (nn - 0.35) * (0.4 + 0.6 * up)
        k = 0.06 + 0.25 * (1 - lit)
        Af = K._ss(th - k, th + k, Ab)
        edge = K._ss(0.02, 0.2, Ab) * (1 - K._ss(0.8, 0.98, Ab))
        Ae = Ae * (1 - edge) + Af * edge
        Ae = np.maximum(Ae, K._ss(0.9, 1.0, Ab))
    out = np.zeros((h, w, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = Ae
    out[..., :3] = K._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


def plume_lobes(rng, X, Yb, Z, width, height, lean=0.1, sun3=None, plumes=None, depth=0.85,
                sizes=(0.16, 0.08, 0.04), dens=1.0, front=0.8, hier=0.8, clump=0.6, flat=0.85, base_cut=None,
                spacing=0.45, neck=0.25, crown=1.25):
    """Cumulus / towering cumulus as a few rising PLUMES (thermals): each plume is a chain of overlapping
    ellipsoids that bulge and neck as it rises (random walk radius), leans downwind and ends in a big
    rounded crown; medium heads and fine cauliflower are grown on the upper / sunward exposed surface.
    plumes: list of (x offset / half width, height fraction, radius fraction, z offset / half width);
    default = main tower + 2-3 lower shoulders. Camera space, px at depth. Returns lobe array (N, 13)."""
    hw = 0.5 * width
    if plumes is None:
        plumes = [(0.0, 1.0, 0.62, 0.0)]
        k = int(rng.integers(2, 4))
        for i in range(k):
            side = -1 if i % 2 == 0 else 1
            plumes.append((side * rng.uniform(0.55, 0.95), rng.uniform(0.3, 0.62), rng.uniform(0.4, 0.55),
                           rng.uniform(-0.5, 0.35)))
    cores = []
    for (xo, hf, rf, zo) in plumes:
        ph_ = rng.uniform(0, 6.28, 2)
        H_ = height * hf
        r0 = hw * rf
        s = 0.0
        rr = r0
        x = X + xo * hw
        z = Z + zo * hw
        while True:
            # radius random walk: bulges and necks; the crown swells
            rr = rr * math.exp(rng.normal(0, neck)) * 0.5 + 0.5 * r0 * (1.0 - 0.3 * s)
            if s > 0.8:
                rr *= 1.0 + (crown - 1.0) * (s - 0.8) / 0.2
            y = Yb - s * H_ - rr * 0.6 * flat
            xx = x + lean * H_ * s + 0.18 * r0 * math.sin(2.1 * math.pi * s + ph_[0])
            if y - rr * flat < Yb - H_ * 1.02:
                y = Yb - H_ + rr * flat * 1.0
                cores.append((np.array([xx, y, z + rng.uniform(-0.1, 0.1) * hw]),
                              np.array([rr * 1.1, rr * flat, rr * depth])))
                break
            cores.append((np.array([xx, y, z + rng.uniform(-0.1, 0.1) * hw]),
                          np.array([rr * rng.uniform(1.0, 1.2), rr * flat, rr * depth])))
            s += rr * spacing / H_
    lob = K.surface_grow(rng, cores, [hw * f for f in sizes], dens=dens, area=math.pi * hw * height * 1.3,
                         sun=sun3, flat=flat, clump=clump, front=front, hier=hier)
    cut = Yb if base_cut is None else base_cut
    rows = []
    for c, r, rt in lob:
        rows.append((c[0], c[1], c[2], r[0], r[1], r[2], -1e30, cut + rng.normal() * 0.003 * height,
                     1.0, 0.0, 0.0, rt, 0.0))
    return np.asarray(rows, np.float64)


def head_lobes(rng, X, Yb, Z, width, height, lean=0.1, sun3=None, n_heads=None, depth=0.85, prof=None,
               r_rng=(0.4, 0.8), recede=0.5, sizes=(0.13, 0.065, 0.032), dens=0.9, front=0.92, hier=0.9,
               clump=0.6, flat=0.82, base_cut=None, wobble=0.12, extra=(), shrink=0.3):
    """Towering cumulus as stacked HEADS: big flattened domes scattered through a bulging envelope,
    lower heads nearer the camera (their lit tops sit in front of the shaded bellies of the heads above),
    then medium / fine cauliflower grown on the exposed upper / sunward surface, mostly on the silhouette.
    prof(s) -> half-width fraction at normalised height s. extra: additional (c, r3) cores."""
    hw = 0.5 * width
    ph = rng.uniform(0, 6.28, 3)
    if prof is None:
        def prof(s):
            return (0.82 - 0.3 * s + 0.1 * math.sin(3.0 * s * math.pi + ph[2])) * math.sqrt(max(1 - s ** 6, 0.05))

    def ax(s):
        return X + lean * height * s + wobble * hw * math.sin(2.2 * math.pi * s + ph[0])

    if n_heads is None:
        n_heads = int(9 + 6 * height / width)
    cores = list(extra)
    # spine (fills the column, never seen on its own)
    ns = max(int(height / (0.3 * hw)), 5)
    for i in range(ns):
        s = (i + 0.5) / ns
        pw = max(prof(s), 0.3) * hw
        r = pw * 0.7
        yy = max(Yb - s * height + r * 0.3, Yb - height + r * flat * 0.9)
        cores.append((np.array([ax(s), yy, Z + recede * s * hw]),
                      np.array([r, r * flat, r * depth])))
    for i in range(n_heads):
        s = 0.97 * rng.random() ** 0.85
        pw = prof(s) * hw
        r = pw * (r_rng[0] + (r_rng[1] - r_rng[0]) * rng.random() ** 1.5) * (1 - shrink * s)
        x = ax(s) + rng.uniform(-1, 1) * max(pw - r * 0.75, 0.0)
        y = Yb - s * height + r * flat * 0.9
        top_ = Yb - height * rng.uniform(0.9, 1.04)
        if y - r * flat < top_:
            r = min(r, 0.5 * (y - top_) / flat + 0.5 * r) if y > top_ else r * 0.6
            y = max(y, top_ + r * flat)
        edge = abs(x - ax(s)) / max(pw, 1e-6)
        z = Z + recede * s * hw - (1 - edge) * 0.35 * r * depth + rng.uniform(-0.1, 0.1) * hw
        cores.append((np.array([x, y, z]), np.array([r * rng.uniform(1.0, 1.2), r * flat * rng.uniform(0.9, 1.05),
                                                    r * depth])))
    lob = K.surface_grow(rng, cores, [hw * f for f in sizes], dens=dens, area=math.pi * hw * height * 1.3,
                         sun=sun3, flat=flat, clump=clump, front=front, hier=hier)
    cut = Yb if base_cut is None else base_cut
    rows = []
    for c, r, rt in lob:
        rows.append((c[0], c[1], c[2], r[0], r[1], r[2], -1e30, cut + rng.normal() * 0.003 * height,
                     1.0, 0.0, 0.0, rt, 0.0))
    return np.asarray(rows, np.float64)


def heap_prof(ph):
    def prof(s):
        return (0.85 + 0.15 * math.sin(math.pi * min(s * 1.5, 1.0) + ph)) * math.sqrt(max(1 - s ** 2.2, 0.03))
    return prof


def vol_cloud_plate(w, h, clouds, sun_dir=(0.75, -0.65), sun_z=-0.1, eye_y=None, persp=4.0, max_dim=260,
                    render=None, paint=None, seed=0, haze=0.0, haze_col=(0.8, 0.88, 0.96), florets=True, L3=None):
    """Volumetric painted cumulus plate. clouds: list of dict(cx, base_y, width, height, kind='tower'|'heap',
    seed, lean, n_heads, sizes, dens, ...) in plate px. Returns straight-alpha RGBA (h, w, 4)."""
    eye = eye_y if eye_y is not None else max(c['base_y'] for c in clouds) + 0.05 * h
    big = max(c['height'] for c in clouds)
    dist = persp * big
    cam = K._side_camera(w, h, eye, dist)
    L = K._sun_vec(None, sun_dir, sun_z, None) if L3 is None else _unit(L3)
    rows = []
    for c in clouds:
        c = dict(c)
        rng = np.random.default_rng(c.pop('seed', 0))
        kind = c.pop('kind', 'tower')
        cx, by, wd, ht = c.pop('cx'), c.pop('base_y'), c.pop('width'), c.pop('height')
        dz = c.pop('dz', 0.0)
        anv = c.pop('anvil', None)
        if anv is not None:
            # flat-topped anvil (wwy_02) in the same lobe language: a wedge of flattened lobes under a flat
            # top plane, a long thin downwind blade, a cone of lobes flaring up from the column
            a = dict(anv)
            top = by - ht - a.pop('rise', 0.0)
            rows.append(anvil_lobes(rng, cx - cam['ppx'] + c.get('lean', 0.0) * ht, top - eye, dist + dz,
                                    a['left'], a['right'], a['thick'], a.get('col_hw', 0.32 * wd),
                                    neck=a.get('neck', 2.2), sun3=L))
        if kind == 'heap':
            c.setdefault('prof', heap_prof(rng.uniform(0, 6.28)))
            c.setdefault('n_heads', int(6 + 5 * wd / max(ht, 1)))
            c.setdefault('recede', 0.3)
            c.setdefault('wobble', 0.05)
        rows.append(head_lobes(rng, cx - cam['ppx'], by - eye, dist + dz, wd, ht, sun3=L, **c))
    E = np.concatenate(rows, 0)
    rk = dict(fine_amp=0.5, erode=(0.8, 8.0, 3, 0.8), seed=seed)
    rk.update(render or {})
    R = vol_render(E, w, h, cam['f'], cam['ppx'], cam['ppy'], L, max_dim=max_dim, **rk)
    pk = dict(k_sun=0.55, seed=seed)
    pk.update(paint or {})
    P = paint_vol(R, w, h, **pk)
    if florets:
        fk = dict(sun_dir=sun_dir, seed=seed)
        fk.update(florets if isinstance(florets, dict) else {})
        edge_florets(P, **fk)
    if haze:
        hc = np.asarray(haze_col, F32)
        P[..., :3] = P[..., :3] * (1 - haze) + hc * haze
    return P


def _billow2(w, h, px, seed):
    """2D billow noise (0..1, round grains ~px across)."""
    from . import core as C
    cells = max(w / max(px * 2.0, 1.0), 2.0)
    n = C.fbm(w, h, cells, 2, seed=seed) * 2 - 1
    return (1.0 - np.abs(n)).astype(F32)


@njit(cache=True)
def _disks(A, C, xs, ys, rs, cols, aa):
    """Rasterise anti-aliased disks: alpha max-union, colour of the winning (last covering) disk."""
    h, w = A.shape
    for i in range(xs.shape[0]):
        x, y, r = xs[i], ys[i], rs[i]
        x0 = max(int(x - r - 2), 0)
        x1 = min(int(x + r + 3), w)
        y0 = max(int(y - r - 2), 0)
        y1 = min(int(y + r + 3), h)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                d = math.sqrt((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2)
                a = min(max((r - d) / aa + 0.5, 0.0), 1.0)
                if a > 0.0:
                    # lobe shading: lit top-sunward crescent, slightly darker underside
                    if a > A[yy, xx]:
                        for c in range(3):
                            C[yy, xx, c] = C[yy, xx, c] * (1 - a) + cols[i, c] * a if A[yy, xx] > 0.5 else cols[i, c]
                        A[yy, xx] = a


def edge_florets(P, sun_dir=(0.75, -0.65), r=(1.5, 9.0), density=0.8, up_min=-0.2, sun_w=0.6, seed=0,
                 inset=(0.35, 0.8), clump=0.8, lift=0.04, r_pow=2.6, gap=(0.7, 2.2)):
    """Fine cauliflower on the silhouette of a painted plate P (RGBA, in place): walks the outer contour and
    adds small overlapping disks where the outline faces up / toward the sun, in clumps; each disk takes the
    plate colour just inside the edge (slightly lifted), so the lit silhouette gets the finest scale of
    lobes while shadow / down sides stay smooth. r: radius range at 1080p (px)."""
    h, w = P.shape[:2]
    sc = w / 1920.0
    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(P[..., 3])
    m = (A > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sd = np.asarray(sun_dir, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    ub = cv2.GaussianBlur(A, (0, 0), 4.0 * sc + 1.0)
    gy, gx = np.gradient(ub)
    xs, ys, rs, cl = [], [], [], []
    ph = rng.uniform(0, 6.28, 2)
    col = P[..., :3]
    for c in cs:
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) < 30:
            continue
        seg = np.sqrt((np.diff(pts, axis=0, append=pts[:1]) ** 2).sum(1))
        sacc = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
        total = float(seg.sum())
        pos = rng.uniform(0, 3.0)
        while pos < total:
            i = min(int(np.searchsorted(sacc, pos)), len(pts) - 1)
            x, y = pts[i]
            xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
            nx, ny = -gx[yi, xi], -gy[yi, xi]
            nn = math.hypot(nx, ny) + 1e-9
            nx, ny = nx / nn, ny / nn
            up = -ny
            face = nx * sd[0] + ny * sd[1]
            fert = 0.5 + 0.5 * math.sin(pos / (50.0 * sc) + ph[0]) * math.sin(pos / (19.0 * sc) + ph[1])
            fert = (1 - clump) + clump * fert
            wgt = np.clip((up - up_min) / (1 - up_min), 0, 1) * (1 - sun_w + sun_w * max(face, 0.0))
            rr = (r[0] + (r[1] - r[0]) * rng.random() ** r_pow) * sc * (0.5 + 0.8 * fert)
            u = rng.random()
            k = rng.uniform(*inset)
            if u < density * wgt * fert * 1.5 and rr > 0.8:
                cx_, cy_ = x - nx * rr * k, y - ny * rr * k
                sx, sy = int(min(max(cx_ - nx * rr * 1.5, 0), w - 1)), int(min(max(cy_ - ny * rr * 1.5, 0), h - 1))
                cc = col[sy, sx] * (1 + lift * max(face, 0.0) + 0.5 * lift)
                xs.append(cx_)
                ys.append(cy_)
                rs.append(rr)
                cl.append(cc)
            pos += rr * rng.uniform(*gap)
    if not xs:
        return P
    C_ = np.ascontiguousarray(P[..., :3])
    _disks(A, C_, np.array(xs), np.array(ys), np.array(rs), np.array(cl, np.float64), 1.0)
    P[..., :3] = C_
    P[..., 3] = A
    return P


# ============================================================================= sea of clouds (deck)
@njit(cache=True, inline='always')
def _h2(ix, iy, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)
    h = h & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / 16777215.0


@njit(cache=True)
def vnoise2(x, y, seed):
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    ix = int(ix)
    iy = int(iy)
    a = _h2(ix, iy, seed)
    b = _h2(ix + 1, iy, seed)
    c = _h2(ix, iy + 1, seed)
    d = _h2(ix + 1, iy + 1, seed)
    return (a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy) * 2.0 - 1.0


@njit(cache=True)
def domes2(x, y, seed, rmin, rmax, keep):
    """Union of random spherical caps (0..1) on a jittered grid: a field of round cumulus puffs."""
    ix = math.floor(x)
    iy = math.floor(y)
    best = 0.0
    for j in range(-1, 2):
        for i in range(-1, 2):
            cx_ = int(ix) + i
            cy_ = int(iy) + j
            if _h2(cx_, cy_, seed + 7) > keep:
                continue
            px = cx_ + _h2(cx_, cy_, seed)
            py = cy_ + _h2(cx_, cy_, seed + 1)
            r = rmin + (rmax - rmin) * _h2(cx_, cy_, seed + 2)
            dx = x - px
            dy = y - py
            q = (dx * dx + dy * dy) / (r * r)
            if q < 1.0:
                v = math.sqrt(1.0 - q) * (0.55 + 0.45 * _h2(cx_, cy_, seed + 3))
                if v > best:
                    best = v
    return best


@njit(cache=True)
def deck_height(X, Z, seed, foot, P):
    """Deck top height at (X, Z). P: params array. foot = pixel footprint (world units) for LOD.
    P = [A_bank, s_bank_x, s_bank_z, A1, s1, A2, s2, A3, s3, A4, s4, gap_amt, gap_s, stretch]"""
    st = P[13]
    # big rolling banks (stretched along X -> horizontal banks at distance)
    b = vnoise2(X / (P[1]), Z / P[2], seed) * 0.6 + vnoise2(X / (P[1] * 0.45), Z / (P[2] * 0.45), seed + 1) * 0.4
    h = P[0] * b
    # dark gaps / valleys
    g = vnoise2(X / P[12], Z / (P[12] * 0.7), seed + 9)
    h -= P[11] * max(g - 0.25, 0.0) * 2.0
    # puffs at 4 scales (sum -> cauliflower on cauliflower), skipped below the pixel footprint
    s = P[4]
    if s > foot * 1.5:
        h += P[3] * domes2(X / (s * st), Z / s, seed + 11, 0.55, 0.95, 0.9)
    s = P[6]
    if s > foot * 1.5:
        h += P[5] * domes2(X / (s * st), Z / s, seed + 23, 0.5, 0.9, 0.85)
    s = P[8]
    if s > foot * 1.5:
        h += P[7] * domes2(X / (s * st), Z / s, seed + 37, 0.5, 0.9, 0.8)
    elif s > foot * 0.5:
        h += P[7] * 0.45
    s = P[10]
    if s > foot * 1.5:
        h += P[9] * domes2(X / s, Z / s, seed + 51, 0.45, 0.85, 0.75)
    elif s > foot * 0.5:
        h += P[9] * 0.4
    return h


@njit(parallel=True, cache=True)
def _deck_render(out, x0, y0, f, ppx, hy, ch, P, seed, Lx, Ly, Lz, soft, sig, sig_l, zmax, g1, g2, gmix,
                 towers, amb_k):
    """out (h, w, 6): sun, ambient, alpha, depth (Z), top-ness, shadow(tower)."""
    hh, ww = out.shape[0], out.shape[1]
    for j in prange(hh):
        for i in range(ww):
            dx = (x0 + i + 0.5 - ppx) / f
            dy = -(y0 + j + 0.5 - hy) / f
            dz = 1.0
            dn = math.sqrt(dx * dx + dy * dy + 1.0)
            dx /= dn
            dy /= dn
            dz /= dn
            if dy >= -1e-5:
                continue
            ytop = P[0] + P[3] + P[5] + P[7] + P[9] + 0.05
            # start where the ray drops below the highest possible top
            t = (ch - ytop) / (-dy)
            if t < 0:
                t = 0.0
            cosT = dx * Lx + dy * Ly + dz * Lz
            ph1 = (1 - g1 * g1) / ((1 + g1 * g1 - 2 * g1 * cosT) ** 1.5)
            ph2 = (1 - g2 * g2) / ((1 + g2 * g2 - 2 * g2 * cosT) ** 1.5)
            ph = ph1 * (1 - gmix) + ph2 * gmix
            T = 1.0
            Ss = 0.0
            Sa = 0.0
            Wz = 0.0
            Wtop = 0.0
            Wsh = 0.0
            jit = _h2(int(i + x0), int(j + y0), seed + 5)
            it = 0
            while T > 0.01 and it < 400:
                it += 1
                foot = max(t, 0.05) / f
                X = dx * t
                Y = ch + dy * t
                Z = dz * t
                if Z > zmax:
                    break
                hgt = deck_height(X, Z, seed, foot, P)
                dd = hgt - Y
                stp = max(0.25 * soft, foot * 1.5, 0.004 * t)
                if dd < -soft:
                    # above the surface: jump (conservative, scaled by the ray slope)
                    t += max((-dd - soft) / (-dy + 0.35) * 0.5, stp)
                    continue
                rho = min(max((dd + soft) / (2.0 * soft), 0.0), 1.0)
                if rho > 0.0:
                    # sun: short march toward the sun through the heightfield (+ tower columns)
                    od = 0.0
                    s_ = 0.03
                    sh = 1.0
                    for k in range(7):
                        qx = X + Lx * s_
                        qy = Y + Ly * s_
                        qz = Z + Lz * s_
                        hq = deck_height(qx, qz, seed, foot * (1 + s_), P)
                        od += min(max((hq - qy + soft) / (2 * soft), 0.0), 1.0) * s_ * 0.75
                        s_ *= 2.0
                    # tower columns (shadow wedges): ray (X,Y,Z)+L*s vs vertical cylinders
                    for m in range(towers.shape[0]):
                        tx, tz, tr, ttop = towers[m, 0], towers[m, 1], towers[m, 2], towers[m, 3]
                        ex = X - tx
                        ez = Z - tz
                        a_ = Lx * Lx + Lz * Lz
                        b_ = 2 * (ex * Lx + ez * Lz)
                        c_ = ex * ex + ez * ez - tr * tr
                        disc = b_ * b_ - 4 * a_ * c_
                        if disc > 0 and a_ > 1e-9:
                            sq = math.sqrt(disc)
                            s1 = (-b_ - sq) / (2 * a_)
                            s2 = (-b_ + sq) / (2 * a_)
                            if s2 > 0:
                                sm = max(s1, 0.0)
                                ym = Y + Ly * sm
                                if ym < ttop:
                                    # soft penumbra from the chord length
                                    ch_ = (s2 - sm) / (2 * tr)
                                    sh = min(sh, 1.0 - min(ch_ * 3.0, 1.0) * 0.92)
                    Ts = math.exp(-sig_l * od) * sh
                    Ta = math.exp(-amb_k * max(dd, 0.0))
                    ext = sig * rho
                    tr_ = math.exp(-ext * stp)
                    w_ = T * (1 - tr_)
                    Ss += w_ * Ts * (ph + 0.25 * math.exp(-sig_l * 0.25 * od) * sh)
                    Sa += w_ * Ta
                    Wz += w_ * Z
                    Wtop += w_ * min(max(1.0 - dd / (6 * soft), 0.0), 1.0)
                    Wsh += w_ * sh
                    T *= tr_
                t += stp * (0.6 + 0.8 * jit) if it == 1 else stp
            A = 1.0 - T
            out[j, i, 0] = Ss
            out[j, i, 1] = Sa
            out[j, i, 2] = A
            if A > 1e-4:
                out[j, i, 3] = Wz / A
                out[j, i, 4] = Wtop / A
                out[j, i, 5] = Wsh / A
            else:
                out[j, i, 3] = zmax


DECK_P = dict(bank=(0.35, 6.0, 3.0), p1=(0.34, 1.3), p2=(0.17, 0.5), p3=(0.08, 0.18), p4=(0.035, 0.065),
              gap=(0.25, 5.0), stretch=1.5)


def deck_fields(w, h, horizon_y, sun, fov=55.0, cam_h=1.0, seed=0, zmax=80.0, soft=0.02, sig=60.0, sig_l=22.0,
                g=(0.6, -0.1), gmix=0.25, lift=0.08, towers=(), amb_k=14.0, box=None, **shape):
    """Ray-march the sea-of-clouds heightfield volume. sun = (x, y) screen px (at infinity).
    towers: list of (X, Z, radius, top_Y) world columns casting shadow wedges.
    Returns dict of (h, w) fields: sun, amb, A, Z, top, shadow; + camera dict."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2)
    ppx = w / 2.0
    L = _unit(((sun[0] - ppx) / f, -(sun[1] - horizon_y) / f + lift, 1.0))
    p = dict(DECK_P)
    p.update(shape)
    P = np.array([p['bank'][0], p['bank'][1], p['bank'][2], p['p1'][0], p['p1'][1], p['p2'][0], p['p2'][1],
                  p['p3'][0], p['p3'][1], p['p4'][0], p['p4'][1], p['gap'][0], p['gap'][1], p['stretch']],
                 np.float64)
    x0, y0, x1, y1 = box if box is not None else (0, int(horizon_y) - 2, w, h)
    out = np.zeros((y1 - y0, x1 - x0, 6), F32)
    T = np.asarray(towers, np.float64).reshape(-1, 4)
    _deck_render(out, float(x0), float(y0), f, ppx, float(horizon_y), cam_h, P, seed, L[0], L[1], L[2], soft,
                 sig, sig_l, zmax, g[0], g[1], gmix, T, amb_k)
    return dict(sun=out[..., 0], amb=out[..., 1], A=out[..., 2], Z=out[..., 3], top=out[..., 4],
                shadow=out[..., 5], box=(x0, y0, x1, y1), f=f, ppx=ppx, L=L)


def paint_deck(R, w, h, horizon_y, sun_x, k_sun=0.36, deep=(0.1, 0.11, 0.26), body=(0.38, 0.4, 0.62),
               sun_col=(1.35, 0.72, 0.3), rim_col=(1.5, 1.05, 0.55), haze_far=(0.74, 0.76, 0.92),
               haze_sun=(1.12, 0.92, 0.74), z_haze=40.0, haze_amt=0.8, near_dark=0.5, near_z=(1.2, 7.0),
               sun_w=0.32, rim_focus=0.55, kuwa=3, seed=0, shadow_col=(0.08, 0.07, 0.22), brush=0.04, near_soft=5.0,
               rim_replace=0.6, near_rim=0.25, patches=0.35):
    """Paint deck fields into an RGB image (h_box, w_box, 3) + alpha + depth."""
    x0, y0, x1, y1 = R['box']
    A = R['A']
    ia = 1.0 / np.maximum(A, 1e-4)
    S = R['sun'] * ia
    Am = np.clip(R['amb'] * ia, 0, 1)
    Z = R['Z']
    sc = w / 1920.0
    hh, ww = A.shape
    xs = (np.arange(x0, x1, dtype=F32)[None, :] - sun_x) / w
    focus = np.exp(-(xs / sun_w) ** 2)
    zf = K._ss(near_z[0], near_z[1] * 3, Z)
    # rims concentrate toward the sun column and the distance (far crests hotter)
    rim_gain = ((1 - rim_focus) + rim_focus * focus) * (near_rim + (1 - near_rim) * zf ** 0.7)
    if patches:
        # sunlight falls in broad patches (gaps in unseen higher cloud): some crest groups light up, others not
        pn = K._noise(ww, hh, max(ww / (420.0 * sc), 2), seed + 7, 3, stretch=3.0)
        rim_gain = rim_gain * np.clip(1 - patches + patches * K._ss(-0.25, 0.35, pn) * 1.3, 0, 1.3)
    s = np.clip(S * k_sun * rim_gain, 0, 2.0)
    if brush:
        s = s * (1 + brush * 4 * K._noise(ww, hh, max(ww / (30.0 * sc), 4), seed + 3, 3, stretch=4, angle=-8))
    deep_ = np.asarray(deep, F32)
    body_ = np.asarray(body, F32)
    col = deep_ + (body_ - deep_) * (Am ** 1.5)[..., None]
    sh = R['shadow'][..., None]
    col = col * (0.55 + 0.45 * sh) + np.asarray(shadow_col, F32) * (1 - sh) * 0.3
    hot = np.clip(s - 0.6, 0, None)
    lit_ = np.clip(s / 0.6, 0, 1)[..., None]
    col = col * (1 - rim_replace * lit_) + np.asarray(sun_col, F32) * lit_ * 0.6 +         np.asarray(rim_col, F32) * hot[..., None]
    # near deck falls into darker navy-violet (yn_02)
    nd = 1 - near_dark * (1 - K._ss(near_z[0], near_z[1], Z))
    col = col * nd[..., None]
    # aerial haze toward the horizon, warmer under the sun
    hz = (1 - np.exp(-Z / z_haze)) * haze_amt
    hc = np.asarray(haze_far, F32) * (1 - focus[..., None]) + np.asarray(haze_sun, F32) * focus[..., None]
    col = col * (1 - hz[..., None]) + hc * hz[..., None]
    # no coverage (rays that ran past zmax): far haze
    cov = K._ss(0.05, 0.6, A)[..., None]
    col = col * cov + hc * (1 - cov)
    if kuwa:
        col = K.kuwahara(col.astype(F32), max(1, int(round(kuwa * sc))), q=6.0)
    if near_soft:
        # airbrushed foreground: the nearest banks lose their detail (soft, painted masses)
        b = cv2.GaussianBlur(col, (0, 0), near_soft * sc)
        wn = (1 - K._ss(near_z[0] * 0.8, near_z[0] * 2.5, Z))[..., None]
        col = col * (1 - wn) + b * wn
    return col.astype(F32)


# ============================================================================= sky-plane cloudlets
@njit(cache=True)
def _cloudlet_density(X, Z, seed, P, foot):
    """Mackerel / altocumulus density on the sky plane. P = [angle, row_l, cell_s, cell_stretch, patch_s,
    patch_thr, warp, cover, lumps]"""
    ca = math.cos(P[0])
    sa = math.sin(P[0])
    u = X * ca + Z * sa
    v = -X * sa + Z * ca
    # rows (streets) with a slow meander
    wv = vnoise2(u / (P[1] * 6.0), v / (P[1] * 6.0), seed + 3) * P[6]
    row = 0.5 + 0.5 * math.sin(2 * math.pi * (v / P[1] + wv))
    # patches: the sky is only partly covered (density gradient, clusters)
    pt = 0.6 * vnoise2(u / P[4], v / (P[4] * 0.6), seed + 5) + 0.4 * vnoise2(u / (P[4] * 0.4), v / (P[4] * 0.3), seed + 6)
    patch = min(max((pt - P[5]) / 0.35, 0.0), 1.0)
    if patch <= 0.0:
        return 0.0
    cs = P[2]
    if cs < foot * 1.2:
        # below the pixel footprint: average coverage (streaky haze band toward the horizon)
        return row * patch * P[7] * 0.55
    c = domes2(u / (cs * P[3]), v / cs, seed + 11, 0.45, 0.8, 0.9)
    if P[8] > 0 and cs * 0.4 > foot * 1.2:
        c = max(c, 0.8 * domes2(u / (cs * 0.45 * P[3]), v / (cs * 0.45), seed + 17, 0.4, 0.8, 0.7))
    # ragged, dry-brush edges
    if cs * 0.25 > foot:
        c += P[8] * 0.3 * vnoise2(u / (cs * 0.22), v / (cs * 0.18), seed + 23)
    if cs * 0.1 > foot:
        c += P[8] * 0.15 * vnoise2(u / (cs * 0.08), v / (cs * 0.07), seed + 29)
    d = c * (0.35 + 0.65 * row) * patch
    return min(max((d - (1.0 - P[7]) * 0.5) * 3.0, 0.0), 1.0)


@njit(parallel=True, cache=True)
def _sky_rows(out, y0, f, ppx, hy, alt, seed, P, sx, sz, lit_d, zmax):
    hh, ww = out.shape[0], out.shape[1]
    for j in prange(hh):
        for i in range(ww):
            dx = (i + 0.5 - ppx) / f
            dy = -(y0 + j + 0.5 - hy) / f
            if dy <= 1e-4:
                continue
            t = alt / dy
            X = dx * t
            Z = t
            if Z > zmax:
                continue
            foot = t / f
            d = _cloudlet_density(X, Z, seed, P, foot)
            if d <= 0.0:
                continue
            e = max(lit_d, foot * 2.0)
            d2 = _cloudlet_density(X + sx * e, Z + sz * e, seed, P, foot)
            d3 = _cloudlet_density(X - sx * e, Z - sz * e, seed, P, foot)
            out[j, i, 0] = d
            out[j, i, 1] = d - d2          # > 0: edge facing the sun
            out[j, i, 2] = d - d3          # > 0: edge facing away
            out[j, i, 3] = Z


def sky_rows_plate(w, h, horizon_y, seed=0, fov=55.0, alt=3.0, angle=25.0, row_l=0.9, cell=0.28,
                   cell_stretch=1.6, patch=6.0, patch_thr=-0.1, warp=0.6, cover=0.55, lumps=1.0,
                   sun=None, sun_dir_plane=(0.3, -1.0), lit_d=0.08, zmax=400.0, region=(0.0, 1.0),
                   lit=(1.02, 0.99, 0.92), body=(0.84, 0.9, 0.98), under=(0.62, 0.74, 0.92), opacity=0.9,
                   haze_col=(0.8, 0.88, 0.97), z_haze=60.0, edge_px=0.8, avoid=None, fade_top=0.0, a_lo=0.05, a_hi=0.35,
                   lit_screen=None, haze_fade=0.6, veil=0.0, veil_px=8.0):
    """Mackerel / altocumulus sky on a perspective plane `alt` above the camera: rippled rows of lumpy
    cloudlets along a flow direction, gathered in patches, larger overhead and compressing into streaky
    bands toward the horizon. Each cloudlet gets a lit edge (facing sun_dir_plane on the plane; (0, -1) =
    toward the camera / lower screen edge) and a soft darker trailing side. Returns straight RGBA (h, w, 4).
    avoid: optional (h, w) float mask (1 = keep clear, e.g. around a tower)."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2)
    y0 = int(max(region[0] * h, 0))
    y1 = int(min(region[1] * h, horizon_y))
    out = np.zeros((h, w, 4), F32)
    if y1 <= y0:
        return out
    buf = np.zeros((y1 - y0, w, 4), F32)
    P = np.array([math.radians(angle), row_l, cell, cell_stretch, patch, patch_thr, warp, cover, lumps], np.float64)
    sd = np.asarray(sun_dir_plane, np.float64)
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    _sky_rows(buf, float(y0), f, w / 2.0, float(horizon_y), alt, seed, P, sd[0], sd[1], lit_d, zmax)
    d, el, eu, Z = buf[..., 0], buf[..., 1], buf[..., 2], buf[..., 3]
    sc = w / 1920.0
    a = K._ss(a_lo, a_hi, d)
    if edge_px:
        a = cv2.GaussianBlur(a, (0, 0), edge_px * sc)
    if veil:
        # soft translucent veil around each shoal (trailing wisps, no cut-out stickers)
        vb = cv2.GaussianBlur(a, (0, 0), veil_px * sc)
        a = np.maximum(a, np.clip(vb * 1.5, 0, 1) * veil)
    if lit_screen is not None:
        # lighting from the (smoothed) coverage in screen space: edge facing lit_screen (px offset) is lit
        db = cv2.GaussianBlur(d, (0, 0), 1.2 * sc + 0.3)
        ox_, oy_ = lit_screen[0] * sc, lit_screen[1] * sc
        el = db - K._shift_img(db, -ox_, -oy_)
        eu = db - K._shift_img(db, ox_, oy_)
    litm = K._ss(0.02, 0.25, el)
    undm = K._ss(0.0, 0.3, eu)
    col = np.asarray(body, F32)[None, None] * np.ones_like(buf[..., :3])
    col = col + (np.asarray(under, F32) - col) * undm[..., None]
    col = col + (np.asarray(lit, F32) - col) * litm[..., None]
    hz = (1 - np.exp(-Z / z_haze))[..., None]
    col = col * (1 - hz) + np.asarray(haze_col, F32) * hz
    a = a * opacity * (1 - haze_fade * hz[..., 0])
    if fade_top:
        yy = np.linspace(0, 1, y1 - y0, dtype=F32)[:, None]
        a = a * (1 - fade_top * (1 - yy))
    out[y0:y1, :, :3] = col
    out[y0:y1, :, 3] = a
    if avoid is not None:
        out[..., 3] *= (1 - avoid)
    out[..., :3] = K._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(2.0, 0.003 * w))
    return out


def strata_plate(w, h, y, thick, seed=0, x_range=(-0.05, 1.05), count=5, length=(0.2, 0.5), opacity=0.75,
                 top=(0.95, 0.97, 0.99), body=(0.76, 0.84, 0.94), under=(0.62, 0.72, 0.88), spread=1.0,
                 streak=14.0):
    """Thin horizontal stratus layers with feathered (torn, tapering) ends and brushed streaks: a lens-shaped
    bar per layer whose alpha is cut by horizontally stretched noise (no hard paper-strip ends), lighter top
    edge, darker flat underside. y / thick in px (layers scatter +-spread*thick around y)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w, 4), F32)
    sc = w / 1920.0
    y0 = int(max(y - thick * (3 + 3 * spread), 0))
    y1 = int(min(y + thick * (3 + 3 * spread), h))
    hh = y1 - y0
    if hh <= 2:
        return out
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(w, dtype=F32)[None, :]
    n = K._noise(w, hh, max(w / (60.0 * sc), 4), seed + 1, 4, stretch=streak)
    n2 = K._noise(w, hh, max(w / (18.0 * sc), 6), seed + 2, 3, stretch=streak * 0.6)
    A = np.zeros((hh, w), F32)
    V = np.zeros((hh, w), F32)
    for i in range(count):
        cx = w * rng.uniform(*x_range)
        L = w * rng.uniform(*length)
        cy = y + rng.uniform(-1, 1) * thick * spread * 1.5
        th = thick * rng.uniform(0.6, 1.3)
        u = (xs - cx) / (L / 2)
        prof = np.clip(1 - np.abs(u) ** 2.2, 0, 1)                   # lens: thick middle, tapering ends
        half = th * (0.25 + 0.75 * prof) * (1 + 0.35 * n)
        dyy = ys - (cy + 0.25 * th * n2)
        inside = (half - np.abs(dyy)) / np.maximum(half, 1e-3)
        a = K._ss(0.0, 0.55, inside) * K._ss(0.0, 0.3, prof + 0.15 * n)
        a = a * np.clip(0.75 + 0.5 * n2, 0, 1)
        v = np.clip(0.5 - dyy / np.maximum(half, 1e-3) * 0.5, 0, 1)   # 1 at the top edge .. 0 at the base
        V = V * (1 - a) + v * a
        A = A + a * (1 - A)
    col = (np.asarray(under, F32) + (np.asarray(body, F32) - np.asarray(under, F32)) * K._ss(0.0, 0.5, V)[..., None])
    col = col + (np.asarray(top, F32) - col) * K._ss(0.6, 0.95, V)[..., None]
    out[y0:y1, :, :3] = col
    out[y0:y1, :, 3] = A * opacity
    out[..., :3] = K._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), 3.0)
    return out


def anvil_lobes(rng, xc, top, Z, left, right, thick, col_hw, neck=2.2, depth=0.9, sun3=None, n=None):
    """Flat-topped anvil (wwy_02): a sheet of flattened lobes under a flat top plane (thickest over the
    column, a long thin downwind blade, a short blunt upwind end), a neck flaring up from the column into
    the sheet (mushroom), cauliflower grown on the underside / edges. Camera space px. Returns (N, 13)."""
    cores = []
    span = left + right
    n = n if n is not None else int(span / (thick * 0.5)) + 6
    for i in range(n):
        u = (i + rng.random()) / n
        x = xc - left + u * span
        uu = (x - xc) / (right if x >= xc else left)
        down = uu > 0
        au = min(abs(uu), 1.0)
        t = thick * ((1 - 0.75 * au ** 1.3) if down else (1 - 0.35 * au ** 2))
        r = t * rng.uniform(0.55, 0.8)
        y = top + r * 0.5
        zj = rng.uniform(-1, 1) * thick * 1.2 * (1 - 0.5 * au)
        cores.append((np.array([x, y, Z + zj]), np.array([r * rng.uniform(1.6, 2.2), r * 0.6, r * 1.4])))
    # neck: the column widening upward into the sheet
    nn = 9
    for k in range(nn):
        s = k / (nn - 1)
        y = top + thick * (0.5 + neck * (1 - s))
        hw = col_hw * (1.0 + 1.8 * s ** 2)
        for side in (-1.0, 0.0, 1.0):
            r = (0.55 + 0.3 * s) * thick * rng.uniform(0.9, 1.2) if side else col_hw * 0.7
            x = xc + side * max(hw - r * 0.6, 0.0)
            cores.append((np.array([x, y, Z + rng.uniform(-0.3, 0.3) * col_hw]),
                          np.array([r * 1.25, r * 0.85, r * depth])))
    lob = K.surface_grow(rng, cores, [thick * 0.3, thick * 0.15, thick * 0.08], dens=0.8,
                         area=span * thick * 3.0, up=(0.0, 1.0, 0.0), front=0.85, down_cut=0.8, flat=0.7,
                         sun=sun3, clump=0.6, hier=0.7)
    rows = []
    for c, r, rt in lob:
        rows.append((c[0], c[1], c[2], r[0], r[1], r[2], top + rng.normal() * 0.01 * thick, 1e30,
                     1.0, 0.0, 0.0, rt, 0.0))
    return np.asarray(rows, np.float64)
