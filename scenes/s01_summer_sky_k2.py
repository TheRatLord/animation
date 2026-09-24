"""s01 helper (round 5): the hero cumulonimbus painted with the shared lib/clouds2 module.

The sun sits at the upper left, just behind the crown's upwind shoulder: the whole left flank and the
crown are a big warm-white lit plane (held at ~92-95 %, never clipped) broken into 2-3 lobed masses with
crisp tops; the shadow is a few large lavender/blue-grey masses tied to the undersides of the lobes
(firm scalloped terminator on the sun side, soft on the far side); cauliflower detail only on the
silhouette; a 2-4 px silver/gold lining on the sun-facing top-left edge; the anvil spreads downwind to the
right and tears into fibres; the base is torn and dissolves into the horizon haze."""
import os
import numpy as np
import cv2

from lib import clouds2 as K

# noon palette pushed from cream toward warm white / ivory (lit ~92-95 %), a clean lavender shadow
PAL = dict(K.PRESETS['noon'])
PAL.update(hi=(0.965, 0.955, 0.93), lit=(0.935, 0.925, 0.9), lit_lo='#d0cfdb', mid='#bdc0d8',
           shade='#9eaadb', deep='#7f8fcb', refl='#b3c1ea', bounce='#d4ccd6', rim=(1.5, 1.45, 1.25),
           haze='#c8e3f6')

V = dict(key=(-0.9, -0.55), sun_z=0.3, width=1.0, anvil_len=0.5, form=0.45, backlit=0.3, inner=1, seed=None, lining=1.7)


def tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8', **kw):
    v = dict(V)
    v.update(kw)
    u = W / 1920.0
    top = base_y - Hc
    # the real sun: just above-left of the crown shoulder (partly hidden behind it as the camera rises)
    sun = (cx - 0.1 * Hc, top + 0.03 * Hc)
    # the key used for the painting: the same sun direction seen from the whole tower (upper left)
    key = (cx + v['key'][0] * Hc, base_y + (v['key'][1] - 1.0) * Hc)
    rgba = K.cumulonimbus_plate(pw, ph, cx, base_y, Hc, sun=key, preset=PAL, seed=v['seed'] or seed, unit=u,
                                sun_z=v['sun_z'], anvil=True, anvil_dir=1.0, anvil_len=v['anvil_len'],
                                width=v['width'], backlit=v['backlit'], form=v['form'], inner=v['inner'], lining=v['lining'],
                                haze_band=(top + 0.62 * Hc, top + 1.0 * Hc, K._c(haze_col), 0.45))
    return _round_steps(rgba, u), np.array(sun, np.float32)


def _round_steps(rgba, u):
    """Paint-out of the stair-stepped (pixel-mask) corners the terminator can leave inside the tower: a
    median filter rounds the corners of flat value planes (a level-set smoothing) while keeping every
    plane boundary crisp. Only the interior is touched (the cauliflower silhouette, its lining and the
    HDR highlights keep their exact paint)."""
    A = rgba[..., 3]
    col = rgba[..., :3]
    k = max(int(round(9 * u)) | 1, 3)
    c8 = np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(c8, k).astype(np.float32) / 255.0
    din = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip((din - 6 * u) / (4 * u), 0, 1) * (col.max(-1) < 0.995)
    w = cv2.GaussianBlur(w.astype(np.float32), (0, 0), 1.0)[..., None]
    out = rgba.copy()
    out[..., :3] = col * (1 - w) + med * w
    return out
