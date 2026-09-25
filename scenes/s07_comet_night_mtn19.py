"""Round-19 mountain repaint for s07_comet_night (run on the range plates before the crisp crest rim).

The mean-shift flattening left hard vector-like snow facets with uniformly crisp edges.  paint_masses:
  * melts the facet boundaries into broad painted value masses (premultiplied blur mixed back in),
  * a warm afterglow wash on the slopes facing the sunset (strongest just under the crest, fading down),
  * a cool violet wash on the slopes turned away,
  * optional soft silhouette (far ranges: hazier, lower-contrast edges -> aerial perspective).
"""
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def paint_masses(pl, s, H, Wp, sun_x, soft=0.5, sigma=4.0, glow=0.3, violet=0.2, edge_soft=0.0, haze=None,
                 haze_amt=0.0):
    rgb, a = pl[..., :3].copy(), np.clip(pl[..., 3], 0, 1)
    h, w = a.shape
    sg = sigma * s + 0.5
    pm = cv2.GaussianBlur(rgb * a[..., None], (0, 0), sg)
    ab = cv2.GaussianBlur(a, (0, 0), sg)[..., None]
    rb = pm / np.maximum(ab, 1e-4)
    k = soft * _ss(0.3, 0.9, ab[..., 0])
    rgb = rgb * (1 - k[..., None]) + rb * k[..., None]
    # per-column crest line + its slope relative to the sun
    top = np.argmax(a > 0.5, axis=0).astype(np.float32)
    top[a.max(0) < 0.5] = h
    tsm = cv2.GaussianBlur(top[None, :], (0, 0), 10 * s + 1)[0]
    slope = np.gradient(tsm)
    xs = np.arange(w, dtype=np.float32)
    toward = np.sign(sun_x - xs)
    face = _ss(-0.3, 0.7, -slope * toward)          # crest rising toward the sun -> slope faces it
    away = _ss(-0.3, 0.7, slope * toward)
    prox = 0.35 + 0.65 * np.exp(-((xs - sun_x) / (0.45 * Wp)) ** 2)
    ys = np.arange(h, dtype=np.float32)[:, None]
    dtop = np.clip(ys - tsm[None, :], 0, None)
    under = np.exp(-dtop / (0.035 * H))
    lum = rgb.max(-1, keepdims=True)
    gw = (under * face[None, :] * prox[None, :] * glow)[..., None]
    rgb = rgb + gw * (0.25 + 0.75 * lum) * np.array([0.9, 0.42, 0.22], np.float32)
    vw = (_ss(0.0, 0.08 * H, dtop) * away[None, :] * violet)[..., None]
    rgb = rgb * (1 - vw) + (rgb * np.array([0.72, 0.66, 1.0], np.float32) + np.array([0.02, 0.0, 0.05], np.float32)) * vw
    if haze is not None and haze_amt > 0:
        rgb = rgb * (1 - haze_amt) + np.asarray(haze, np.float32) * haze_amt
    if edge_soft > 0:
        a = cv2.GaussianBlur(a, (0, 0), edge_soft * s + 0.3)
    return np.dstack([np.clip(rgb, 0, None), a]).astype(np.float32)
