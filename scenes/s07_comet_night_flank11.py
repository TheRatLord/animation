"""Round-11 painted flank structure for the near snow ranges of s07_comet_night.

The ray-cast terrain leaves the long shoulders of the near massifs as broad flat snow planes.  This pass
over-paints them in image space the way a background painter would:
  * fall-line coordinate: every column is traced down from the skyline along the ridge normal, so the
    structure runs down the slope, not straight down the screen,
  * couloirs: narrow, meandering snow-filled channels (ridged 1-D noise across the fall line) separated by
    darker exposed-rock ribs; each channel wall turned toward the comet carries a CRISP lit edge, the wall
    turned away falls off into a cool, lost shadow,
  * rock strata: ledges running parallel to the skyline, dark risers broken where couloirs cut through,
  * big darker rock masses (low-frequency) so the value structure is not one flat pale plane,
  * a bright rim hugging the skyline on the comet side (thicker near the summits).
Returns a new straight RGBA plate.
"""
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _sm1(a, sigma):
    return cv2.GaussianBlur(np.asarray(a, np.float32).reshape(1, -1), (0, 0), sigmaX=max(sigma, 0.5),
                            borderType=cv2.BORDER_REFLECT)[0]


def paint_flanks(rgba, light_x, Wp, H, s, seed=0, region=None, amt=1.0):
    rgb = rgba[..., :3].copy()
    a = rgba[..., 3]
    Hh, Ww = a.shape
    xr = np.arange(Ww, dtype=np.float32)
    on = a > 0.5
    has = on.any(0)
    ytop = np.where(has, on.argmax(0), Hh).astype(np.float32)
    ytop_s = _sm1(np.where(has, ytop, np.nan_to_num(ytop)), 0.012 * Wp)
    g = np.clip(np.gradient(_sm1(ytop, 0.03 * Wp)), -1.5, 1.5)
    xs, ys = C.grid(Ww, Hh)
    dd = np.clip(ys - ytop_s[None, :], 0, None)
    depth = 0.2 * H
    # fall-line coordinate (constant along lines running down the ridge normal), gently meandering
    mea = P.fbm1d((dd / (0.08 * H)).ravel(), 1.0, 3, seed + 3).reshape(dd.shape)
    xc = xs + 1.2 * g[None, :] * dd + 6.0 * s * mea
    # anisotropic painted fields sampled in (fall-line, depth) space: long along the fall line
    pad = int(0.5 * Ww)
    mx = (xc + pad).astype(np.float32)
    my = (dd * 0.3).astype(np.float32)

    def field(scale_px, octv, sd):
        T = C.fbm(Ww + 2 * pad, Hh, (Ww + 2 * pad) / (scale_px * s), octv, seed=sd).astype(np.float32)
        return cv2.remap(T, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    Fm = field(65.0, 4, seed + 5)
    Ff = field(14.0, 3, seed + 7)
    G = 1 - np.abs(2 * Fm - 1)                            # 1 on channel lines
    G2 = 1 - np.abs(2 * Ff - 1)
    # only some channels, each fading out at its own depth
    pres = field(90.0, 2, seed + 9)
    reach = (0.25 + 0.9 * pres) * depth
    gul = _ss(0.7, 0.88, G) * _ss(0.4, 0.55, pres) * _ss(reach, reach * 0.5, dd)
    gul = np.clip(gul + 0.5 * _ss(0.86, 0.97, G2) * _ss(0.45, 0.6, pres) * _ss(reach * 0.6, 0.0, dd), 0, 1)
    # channel walls: which side faces the comet
    dG = np.gradient(cv2.GaussianBlur((0.8 * G + 0.2 * G2).astype(np.float32), (0, 0), 0.8), axis=1)
    side = np.sign(light_x - xs)
    facing = dG * side
    wall_lit = _ss(0.004, 0.02, facing)
    wall_sh = _ss(0.0, -0.03, facing)
    # big exposed-rock masses, elongated down the fall line
    big = field(70.0, 4, seed + 11)
    rib = _ss(0.56, 0.68, big + 0.1 * np.exp(-dd / (0.05 * H))) * (1 - gul)
    # strata inside the rock masses: ledges parallel to the skyline
    wob = C.fbm(Ww, Hh, Ww / (0.04 * H), 3, seed=seed + 13)
    ph = (dd + 14.0 * s * (wob - 0.5) * 2) / (26.0 * s) + 0.002 * xs
    fr = ph - np.floor(ph)
    riser = _ss(0.0, 0.08, fr) * _ss(0.45, 0.25, fr) * rib * _ss(0.01 * H, 0.03 * H, dd)

    lit_snow = np.array([0.66, 0.78, 1.0], np.float32)
    snow = np.array([0.42, 0.52, 0.86], np.float32)
    sh_snow = np.array([0.24, 0.29, 0.62], np.float32)
    rock_d = np.array([0.07, 0.09, 0.25], np.float32)
    rock_l = np.array([0.2, 0.25, 0.52], np.float32)
    # base: keep the terrain's own value, pushed toward painted steps
    lum = rgb.mean(-1)
    base = rgb.copy()
    # couloir snow: bright tongue, lit wall crisp, shadow wall lost
    c_snow = snow * (1 - wall_lit[..., None]) + lit_snow * wall_lit[..., None]
    c_snow = c_snow * (1 - 0.6 * wall_sh[..., None]) + sh_snow * 0.6 * wall_sh[..., None]
    out = base * (1 - gul[..., None] * 0.75) + c_snow * (gul[..., None] * 0.75)
    # rock ribs: dark masses with a lit face toward the comet
    rk = rock_d * (1 - wall_lit[..., None] * 0.7) + rock_l * (wall_lit[..., None] * 0.7)
    rb = rib[..., None] * 0.75
    out = out * (1 - rb) + rk * rb
    out = out * (1 - 0.5 * riser[..., None]) + rock_d * 0.5 * riser[..., None]
    # snow ledge tops just above each riser catch the light
    ledge = _ss(0.45, 0.55, fr) * _ss(0.9, 0.7, fr) * rib * _ss(0.01 * H, 0.03 * H, dd)
    out = out + ledge[..., None] * (snow - out) * 0.3
    # weight: flanks only (below the skyline rim, fading out low down where the forest takes over)
    w = _ss(2.0 * s, 8.0 * s, dd) * _ss(depth, 0.35 * depth, dd) * on
    if region is not None:
        w = w * region(xr)[None, :]
    w = w * amt * _ss(0.1, 0.25, lum)
    rgb = rgb * (1 - w[..., None]) + out * w[..., None]
    # skyline rim on the comet side, thicker near summits (high skyline = small ytop)
    hi = _ss(np.percentile(ytop[has], 60), np.percentile(ytop[has], 5), ytop)[None, :]
    face = _ss(-0.05, 0.25, -g * np.sign(light_x - xr))[None, :]
    rimw = (1.2 + 2.2 * hi) * s
    rim = np.exp(-dd / rimw) * on * (0.35 + 0.65 * face) * (0.5 + 0.5 * hi)
    if region is not None:
        rim = rim * region(xr)[None, :]
    rgb = rgb + rim[..., None] * np.array([0.55, 0.75, 0.95], np.float32) * 0.9 * amt
    return np.dstack([rgb, a]).astype(np.float32)
