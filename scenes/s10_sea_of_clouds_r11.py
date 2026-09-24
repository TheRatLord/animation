"""s10 helpers (round 11): a sea of BROKEN cumulus tops painted with lib.clouds2, split into depth plates.

Every top is an individual clouds2.cumulus (a few flat painted masses sharing one light: warm-gold lit
crown toward the sun, rose terminator, cool blue-grey shadow side, crisp cauliflower detail only on the
silhouette, torn base). Tops are scattered on a ground plane in perspective: screen size is proportional
to the distance below the horizon, so they shrink, pale and lose contrast toward a hazy gold horizon.
Gaps between them reveal a deep blue under-floor. The field is split by depth into plates that the scene
moves at different rates (true multi-plane parallax for the dolly / crane).

Plate = dict(rgba, ox, oy, d): straight-alpha RGBA whose pixel (0,0) sits at frame (ox, oy) at rest;
d = parallax factor (0 horizon .. 1 nearest tops).
"""
import math

import numpy as np
import cv2

from lib import clouds2 as K

# ----------------------------------------------------------------------------------------- palettes
# near tops: warm gold crowns, rose terminator, cool blue-grey shadow, deep blue valleys
GOLD = dict(hi=(1.06, 0.97, 0.8), lit='#ffd08a', lit_lo='#f6ac8a', mid='#e08e98', shade='#7489c2',
            deep='#2f4590', refl='#93a8dc', bounce='#e6a896', rim=(1.5, 1.16, 0.72), haze='#f4c8ae',
            edge_dark=0.06)
# off the sun axis: pinker, cooler lit faces
ROSE = dict(hi=(1.0, 0.9, 0.82), lit='#fbc0a2', lit_lo='#eda2a4', mid='#d68ca8', shade='#7486c4',
            deep='#2f4592', refl='#94a6dc', bounce='#e0a4a4', rim=(1.34, 1.06, 0.8), haze='#f0c4bc',
            edge_dark=0.06)
# toward the horizon: pale, low contrast, gold haze
FAR = dict(hi=(1.0, 0.93, 0.82), lit='#fcd6b4', lit_lo='#f4c0b0', mid='#e8b0b6', shade='#b4b6da',
           deep='#9ea6d4', refl='#bcc4e4', bounce='#eeb8b0', rim=(1.2, 1.04, 0.84), haze='#f8d8bc',
           edge_dark=0.03)
# towers (sunrise, lit from the side / slightly behind)
TOWER = dict(hi=(1.04, 0.96, 0.84), lit='#ffe0b8', lit_lo='#f6bca6', mid='#e29aa8', shade='#8a92cc',
             deep='#4c5aa6', refl='#a4ade0', bounce='#f2b4a0', rim=(1.55, 1.2, 0.76), haze='#f4d0c0',
             edge_dark=0.05)
TOWER_FAR = dict(hi=(1.0, 0.95, 0.86), lit='#fcdcc0', lit_lo='#f2c2b4', mid='#e4aab8', shade='#a6a8d8',
                 deep='#7e86c4', refl='#b6bce4', bounce='#eebcb0', rim=(1.35, 1.1, 0.82), haze='#f6d6c6',
                 edge_dark=0.04)


def floor_color(dy):
    """Colour of the deep under-floor seen in the gaps, by depth below the horizon (fraction of H)."""
    k = np.clip(dy / 0.5, 0, 1) ** 0.6
    far = np.array(K._c('#e8c2c0'), np.float32)
    mid = np.array(K._c('#7a86c4'), np.float32)
    near = np.array(K._c('#384a98'), np.float32)
    a = np.clip(k / 0.35, 0, 1)[..., None]
    b = np.clip((k - 0.35) / 0.65, 0, 1)[..., None]
    return (far * (1 - a) + mid * a) * (1 - b) + near * b


# ----------------------------------------------------------------------------------------- field layout
def layout(W, H, hy, sx, seed=3, dy_min=0.006, dy_max=0.8, x_margin=0.16, gap=0.2, S=0.72):
    """Scatter cumulus tops on the ground plane. Returns a list of dicts sorted far -> near:
    cx, by (base y in frame px), w, h, dy (base depth below horizon, fraction of H), prox (sun axis), seed."""
    rng = np.random.default_rng(seed)
    out = []
    dy = dy_min
    k = 0
    while dy < dy_max:
        step = dy * rng.uniform(0.2, 0.34) + 0.002
        cw0 = S * W * dy                      # typical screen width at this depth
        x = -x_margin * W - rng.uniform(0, 1) * cw0
        while x < (1 + x_margin) * W + cw0:
            r = float(np.clip(math.exp(rng.normal(0.0, 0.35)), 0.55, 1.7))
            cw = cw0 * r
            # breathing room: skipped slots reveal the deep blue floor (fewer at the horizon)
            if rng.random() > gap * min(dy / 0.08, 1.0):
                ddy = dy + rng.uniform(-0.55, 0.55) * step
                asp = rng.uniform(0.34, 0.56)
                cx = x + cw * 0.5
                prox = math.exp(-((cx - sx) / (0.22 * W + 0.6 * W * ddy)) ** 2)
                if rng.random() < 0.12 and 0.05 < ddy < 0.3 and prox < 0.3:
                    asp *= rng.uniform(1.2, 1.45)         # an occasional heaped turret (never under the sun)
                if ddy < 0.25:
                    asp *= 1.0 - 0.4 * prox               # keep the horizon under the sun open
                # far tops look flatter (seen edge-on through haze they merge into a sheet)
                asp *= 0.75 + 0.25 * min(ddy / 0.1, 1.0)
                # near tops are seen from above: lower domes, never towering over the horizon
                asp *= 1.0 - 0.35 * min(max(ddy - 0.15, 0.0) / 0.3, 1.0)
                out.append(dict(cx=cx, by=hy + ddy * H, w=cw, h=cw * asp, dy=ddy, prox=prox, seed=int(k)))
                k += 1
            x += cw * rng.uniform(0.42, 0.78)
        dy += step
    out.sort(key=lambda c: c['by'])
    return out


def paint_band(W, H, hy, sun, tops, dy0, dy1, pad=0.12, seed=0, lining=1.0, sun_z=-0.05, crop=None):
    """Paint the tops whose base depth is in [dy0, dy1) on one plate (with margins). Returns a plate."""
    sel = [c for c in tops if dy0 <= c['dy'] < dy1]
    if not sel:
        return None
    u = W / 1920.0
    x0 = -pad * W
    y_top = min(c['by'] - c['h'] * 1.6 for c in sel)
    y_bot = max(c['by'] + 0.12 * c['h'] for c in sel)
    y0 = max(int(y_top - 8 * u), int(-pad * H))
    y1 = min(int(y_bot + 8 * u), int(H * (1 + pad)))
    pw, ph = int((1 + 2 * pad) * W), max(int(y1 - y0), 8)
    sp = (sun[0] - x0, sun[1] - y0)
    P = K.Painter(pw, ph, sun=sp, pal=GOLD, unit=u, seed=seed + 5, sun_z=sun_z)
    for c in sel:
        f = float(np.clip(c['dy'] / 0.12, 0, 1))          # 0 horizon .. 1 mid-distance and nearer
        near = float(np.clip((c['dy'] - 0.12) / 0.4, 0, 1))
        pal = K.mix_palette(ROSE, GOLD, 0.25 + 0.75 * c['prox'])
        pal = K.mix_palette(FAR, pal, K._ss(0.0, 1.0, f) ** 0.8)
        if near > 0:
            # nearest tops: deeper, cooler shadow side (value range grows toward the camera)
            pal = dict(pal)
            dv = np.asarray(K._c('#4e62a8'), np.float32)
            pal['shade'] = (pal['shade'] * (1 - 0.55 * near) + dv * 0.55 * near).astype(np.float32)
        # torn bases sink into the valley colour: pale gold haze far away, deep blue near the camera
        if f > 0.3:
            pal = dict(pal)
            kd = float(K._ss(0.3, 1.0, f)) * (0.6 + 0.4 * near)
            pal['haze'] = (pal['haze'] * (1 - kd) + np.asarray(K._c('#3e519c'), np.float32) * kd).astype(np.float32)
        haze = 0.55 * (1 - f) ** 1.5
        n = 2 if c['w'] < 40 * u else (3 if c['w'] < 260 * u else 4)
        lv = None
        if c['w'] < 60 * u:
            lv = ((0.06, 0.14, 0.85, 1.1, 2.0), (0.025, 0.05, 0.6))
        K.cumulus(P, c['cx'] - x0, c['by'] - y0, c['w'], c['h'], seed=1000 + c['seed'], pal=pal, n=n, haze=haze,
                  wisps=c['w'] > 30 * u, levels=lv, lost=0.03 + 0.08 * (1 - f), sun_jitter=12.0,
                  sun_z=sun_z - 0.22 * near)
    if lining:
        P.lining(lining, rim_px=2.2 + 1.2 * min(dy0 / 0.2, 1.0), halo=0.6)
    return dict(rgba=P.rgba(), ox=x0, oy=y0, d=None)


def place(img, plate, z, pivot, tx, ty, W, H):
    """'Over' composite a plate zoomed by z about pivot and translated by (tx, ty); returns sampled alpha."""
    rgba = plate['rgba']
    px, py = pivot
    M = np.array([[z, 0, z * plate['ox'] + px * (1 - z) + tx],
                  [0, z, z * plate['oy'] + py * (1 - z) + ty]], np.float32)
    s = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = np.clip(s[..., 3:4], 0, 1)
    img *= (1 - a)
    img += s[..., :3] * a
    return a[..., 0]


# ----------------------------------------------------------------------------------------- fast compositing
from numba import njit, prange  # noqa: E402


@njit(cache=True, fastmath=True, parallel=True)
def _over(img, src, y0, x0, aout):
    h, w = src.shape[0], src.shape[1]
    for i in prange(h):
        for j in range(w):
            a = src[i, j, 3]
            if a <= 0.0:
                continue
            if a > 1.0:
                a = 1.0
            y, x = y0 + i, x0 + j
            for c in range(3):
                img[y, x, c] = img[y, x, c] * (1.0 - a) + src[i, j, c] * a
            aout[y, x] = a


def place_fast(img, plate, z, pivot, tx, ty, W, H, alpha=None, amul=1.0):
    """Like place(), but warps only the plate's on-screen bounding box and composites with numba.
    Returns the sampled alpha as a full (H, W) array when `alpha` is given (written into it)."""
    rgba = plate['rgba']
    ph, pw = rgba.shape[:2]
    px, py = pivot
    ax = z * plate['ox'] + px * (1 - z) + tx
    ay = z * plate['oy'] + py * (1 - z) + ty
    X0, Y0 = max(int(math.floor(ax)) - 1, 0), max(int(math.floor(ay)) - 1, 0)
    X1, Y1 = min(int(math.ceil(ax + z * pw)) + 1, W), min(int(math.ceil(ay + z * ph)) + 1, H)
    if X1 <= X0 or Y1 <= Y0:
        return alpha
    M = np.array([[z, 0, ax - X0], [0, z, ay - Y0]], np.float32)
    s = cv2.warpAffine(rgba, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
    if amul != 1.0:
        s[..., 3] *= np.float32(amul)
    aout = alpha if alpha is not None else np.zeros((H, W), np.float32)
    _over(img, s, Y0, X0, aout)
    return aout


@njit(cache=True, fastmath=True, parallel=True)
def _finish(x, paper, vig, exposure, sat):
    H, W = x.shape[0], x.shape[1]
    out = np.empty_like(x)
    for i in prange(H):
        for j in range(W):
            r = max(x[i, j, 0] * exposure, 0.0)
            g = max(x[i, j, 1] * exposure, 0.0)
            b = max(x[i, j, 2] * exposure, 0.0)
            m = max(r, max(g, b))
            s0, k = 0.84, 0.16
            mc = m
            if m > s0:
                o = m - s0
                mc = s0 + k * o / (o + k)
            f = mc / max(m, 1e-6)
            wd = min(max((m - 1.6) / 3.4, 0.0), 1.0) * 0.9
            pp = paper[i, j, 0]
            r = (r * f * (1 - wd) + mc * wd) * pp
            g = (g * f * (1 - wd) + mc * wd) * pp
            b = (b * f * (1 - wd) + mc * wd) * pp
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            v = vig[i, j, 0]
            out[i, j, 0] = min(max((lum + (r - lum) * sat) * v, 0.0), 1.0)
            out[i, j, 1] = min(max((lum + (g - lum) * sat) * v, 0.0), 1.0)
            out[i, j, 2] = min(max((lum + (b - lum) * sat) * v, 0.0), 1.0)
    return out


def finish(x, paper, vig, exposure, sat):
    return _finish(np.ascontiguousarray(x, dtype=np.float32), paper, vig, np.float32(exposure), np.float32(sat))


@njit(cache=True, fastmath=True, parallel=True)
def _path(img, path, near_a, gain, fgshade):
    """In place: specular glow path keyed on the lit cloud tops (stronger on the nearest plane) + the
    foreground's slight darkening."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            x = min(max((r - 0.7) / 0.3, 0.0), 1.0)       # warm lit crowns only (not the blue bodies)
            key = x * x * (3 - 2 * x) * (0.6 + 0.4 * near_a[i, j]) * gain
            img[i, j, 0] = (r + path[i, j, 0] * key) * fgshade[i, 0, 0]
            img[i, j, 1] = (g + path[i, j, 1] * key) * fgshade[i, 0, 1]
            img[i, j, 2] = (b + path[i, j, 2] * key) * fgshade[i, 0, 2]


def glow_path(img, path, near_a, gain, fgshade):
    _path(img, path, near_a, np.float32(gain), fgshade)
    return img
