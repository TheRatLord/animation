"""Multi-plane painted cloud sea for s10.

The sea of clouds is built like an anime background: rows of painted clouds (lib.sky.cumulus_plate, the
dome engine: crisp cauliflower lobes, gold rims, terminator bands) standing at increasing depths z in
front of the camera.  Every row is a flat plate in world units (x along the row, height above the deck)
rendered once at the resolution it needs at its closest; each frame it is placed with exact pinhole
perspective (screen scale f / z), so the forward glide gives true multi-plane parallax: near rows sweep
down and grow, far rows barely move.  Behind each row's clouds sits a soft 'deck' band (the cloud layer
continuing below the tops) so gaps show deeper cloud instead of sky.
"""
import math
import numpy as np
import cv2

from lib import core as C, sky as S, clouds as K

# Child lobes grown toward the viewer read as face-on 'polka dots' when the cloud is backlit.  For this
# shot the lobes are grown on the crowns / silhouettes instead (growth directions turned away from the
# camera), so front faces stay large coherent billows and the silhouettes carry the cauliflower detail.
FRONT = {1: (-0.45, 0.3), 2: (-0.6, 0.05), 3: (-0.6, 0.0)}
# smaller, more numerous crown bumps (cauliflower texture instead of a few round beads)
SUBDIV = {1: (1.15, 0.9), 2: (2.2, 0.62)}


class crown_lobes:
    """Context manager: temporarily route lib.clouds child-lobe growth through FRONT (restored after)."""

    def __enter__(self):
        self._orig = K._children
        orig = self._orig

        def ch(B, rng, P, cid, lev, count, ratio, front, *a, **k):
            cm, rm = SUBDIV.get(lev, (1.0, 1.0))
            return orig(B, rng, P, cid, lev, count * cm, ratio * rm, FRONT.get(lev, front), *a, **k)
        K._children = ch
        return self

    def __exit__(self, *exc):
        K._children = self._orig
        return False


def row_specs(rng, x0u, x1u, hmax, big=0.18, strato=0.6, wscale=1.0, gap=(0.3, 0.55)):
    """Clouds along a row, in world units: list of (cx, base_off, w, h, kind, flat)."""
    out = []
    x = x0u + rng.uniform(-1.0, 0.0)
    while x < x1u + 1.0:
        isbig = rng.random() < big
        w = rng.uniform(1.4, 3.6) * wscale * (1.6 if isbig else 1.0)
        if isbig or rng.random() > strato:
            kind = 'cumulus'
            h = w * rng.uniform(0.3, 0.48)
        else:
            kind = 'stratocumulus'
            h = w * rng.uniform(0.17, 0.26)
        h = min(h, hmax * rng.uniform(0.8, 1.0))
        out.append((x, rng.uniform(-0.06, 0.06), w, h, kind, rng.uniform(0.8, 1.0)))
        x += w * rng.uniform(gap[0], gap[1])
    return out


def build_row(seed, z_ref, ppu, x0u, x1u, hmax, f, W, hy, camx, camy, sun_scr, pal, deck_top, deck_bot,
              haze_fn, fog, backlit=0.5, big=0.18, strato=0.6, wscale=1.0, blur_px=0.0, rim=1.3, sun_z=-0.2,
              detail=1.0, lit_bias=0.0, deck=True, gap=(0.3, 0.55), soft=0.16):
    """Render one row plate.

    z_ref: depth used to map the on-screen sun into the plate (lighting); ppu: plate px per world unit;
    x0u..x1u: world x extent; hmax: tallest cloud (world units); sun_scr: (x, y) sun on screen;
    deck_top/deck_bot: colours of the deck band under the row; haze_fn(xs_screen) -> (n, 3) haze colour;
    fog: 0..1 aerial perspective of this row.
    Returns dict(rgba, ppu, x0u, top) where the plate's y = top + (hmax - h) * ppu for world height h."""
    rng = np.random.default_rng(seed)
    top = 0.45 * ppu
    base_py = top + hmax * ppu
    under = 1.2 * ppu
    pw = int(math.ceil((x1u - x0u) * ppu))
    ph = int(math.ceil(base_py + under))
    specs = []
    for (cxu, bo, wu, hu, kind, flat) in row_specs(rng, x0u, x1u, hmax, big, strato, wscale, gap):
        specs.append(((cxu - x0u) * ppu, base_py + bo * ppu, wu * ppu, hu * ppu, 2.0, kind, flat))
    # on-screen sun -> plate coordinates at z_ref
    Xs = camx + (sun_scr[0] - W / 2) * z_ref / f
    hs = camy - (sun_scr[1] - hy) * z_ref / f
    sun_p = ((Xs - x0u) * ppu, top + (hmax - hs) * ppu)
    # the screen width of this plate at z_ref (for the backlit radius, expressed in plate widths)
    a = f / (z_ref * ppu)
    br = 0.22 * W / (a * pw)
    # clouds only need the band above the base (+ a little for the torn skirts); the deck fills the rest
    phc = min(ph, int(math.ceil(base_py + 0.3 * ppu)))
    sky = np.broadcast_to(np.asarray(deck_top, np.float32), (phc, pw, 3))
    clc = S.cumulus_plate(pw, phc, seed=seed, clouds=specs, palette=pal, sun_pos=sun_p, sun_z=sun_z,
                          backlit=backlit, backlit_radius=br, rim=rim, depth_fog=0.0, sky=sky, halo=0.3,
                          texture=0.6, detail=detail, lit_bias=lit_bias, soft=soft)
    cl = np.zeros((ph, pw, 4), np.float32)
    cl[:phc] = clc
    # deck band under the tops: soft torn upper edge, darker downward
    ys = np.arange(ph, dtype=np.float32)[:, None]
    xs = np.arange(pw, dtype=np.float32)[None, :]
    edge = base_py - 0.22 * ppu + 0.08 * ppu * np.sin(xs / (0.9 * ppu) + seed) + \
        0.05 * ppu * np.sin(xs / (0.31 * ppu) + 2 * seed)
    da = C.smoothstep(edge - 0.1 * ppu, edge + 0.25 * ppu, ys) * (1.0 if deck else 0.0)
    g = C.smoothstep(base_py - 0.3 * ppu, ph, ys)[..., None]
    dcol = C.lerp(np.asarray(deck_top, np.float32), np.asarray(deck_bot, np.float32), g)
    dcol = np.broadcast_to(dcol, (ph, pw, 3))
    ca = cl[..., 3:4]
    alpha = ca + da[..., None] * (1 - ca)
    rgb = (cl[..., :3] * ca + dcol * da[..., None] * (1 - ca)) / np.maximum(alpha, 1e-5)
    # aerial perspective toward the haze colour on screen
    xs_scr = W / 2 + (f / z_ref) * (x0u + (np.arange(pw) + 0.5) / ppu - camx)
    hz = haze_fn(xs_scr)[None, :, :]
    rgb = C.lerp(rgb, hz, fog)
    rgba = np.dstack([rgb, alpha[..., 0]]).astype(np.float32)
    if blur_px > 0.3:
        pm = rgba.copy()
        pm[..., :3] *= pm[..., 3:4]
        pm = cv2.GaussianBlur(pm, (0, 0), blur_px)
        rgba = np.dstack([pm[..., :3] / np.maximum(pm[..., 3:4], 1e-5), pm[..., 3]]).astype(np.float32)
    return dict(rgba=np.ascontiguousarray(rgba), ppu=ppu, x0u=x0u, top=top, hmax=hmax)


def place_row(R, z, f, W, H, hy, camx, camy, xshift=0.0):
    """Affine plate->screen for a row at depth z. Returns (M, y0) where y0 = first screen row covered."""
    a = f / (z * R['ppu'])
    bx = W / 2 + (f / z) * (R['x0u'] - camx) + xshift
    by = hy + (f / z) * (camy - R['hmax']) - a * R['top']
    return np.array([[a, 0, bx], [0, a, by]], np.float32), by


def render_row(R, z, f, W, H, hy, camx, camy, xshift=0.0):
    """Sample the row into the frame. Returns (y0, rgba (H - y0, W, 4)) or None if off-screen."""
    M, by = place_row(R, z, f, W, H, hy, camx, camy, xshift)
    y0 = int(max(math.floor(by), 0))
    if y0 >= H:
        return None
    M = M.copy()
    M[1, 2] -= y0
    out = cv2.warpAffine(R['rgba'], M, (W, H - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return y0, out


def footprint(R, blur_u=0.08):
    """Per plate column: world height of the cloud top (0 where there is no cloud), softened a little.
    Used to cast the heads' shadows onto the deck."""
    a = R['rgba'][..., 3] > 0.4
    has = a.any(0)
    y = np.argmax(a, 0).astype(np.float32)
    h = np.where(has, R['hmax'] - (y - R['top']) / R['ppu'], 0.0).astype(np.float32)
    h = np.maximum(h, 0.0)
    s = blur_u * R['ppu']
    if s > 0.5:
        h = cv2.GaussianBlur(h[None, :], (0, 0), sigmaX=s, sigmaY=0.1)[0]
    return np.ascontiguousarray(h, dtype=np.float32)
