"""s04_rain_street: cropped, single-surface plates ("cards") for signs and standees.

Every projecting sign lives on its own small plate with one smooth depth, so the push-in re-projection never
folds or smears glyphs over depth discontinuities (sign vs. wall behind vs. neighbouring sign) and parallax
between overlapping signs reveals real content instead of stretched pixels. Cards are baked from a crop of
the supersampled canvas and warped per frame only inside their projected bounding box."""
import copy
from types import SimpleNamespace

import numpy as np
import cv2

import s04_rain_street_r as R


def bake_card(cv, lights, ambient, fog=None, dof=None, glow=None, pad=0.06, name='card', mirror=True):
    """Bake the non-empty part of canvas `cv` into a cropped Plate (+ cropped mirror). Returns the plate or
    None. plate.ox/oy = crop offset in 1x plate pixels."""
    cam = cv.cam
    s = cam.ss
    rows = np.flatnonzero(cv.a.max(1) > 1e-3)
    if rows.size == 0:
        return None
    cols = np.flatnonzero(cv.a[rows[0]:rows[-1] + 1].max(0) > 1e-3)
    pp = int(pad * cam.W)
    x0 = max(int(cols[0]) // s - pp, 0)
    x1 = min(int(cols[-1]) // s + 1 + pp, cam.PW)
    y0 = max(int(rows[0]) // s - pp, 0)
    y1 = cam.PH if mirror else min(int(rows[-1]) // s + 1 + pp, cam.PH)
    cc = copy.copy(cam)
    cc.pcx = cam.pcx - x0
    cc.pcy = cam.pcy - y0
    cc.PW, cc.PH = x1 - x0, y1 - y0
    sl = (slice(y0 * s, y1 * s), slice(x0 * s, x1 * s))
    sub = SimpleNamespace(cam=cc, alb=cv.alb[sl], emi=cv.emi[sl], a=cv.a[sl], z=cv.z[sl], n=cv.n[sl])
    if fog is not None:
        fc = np.asarray(fog[0], np.float32)
        if fc.ndim == 3:
            fc = fc[y0:y1, x0:x1]
        fog = (fc,) + tuple(fog[1:])
    p = R.bake(sub, lights, ambient, fog=fog, dof=dof, name=name, glow=glow)
    p.ox, p.oy = x0, y0
    if mirror:
        p.make_mirror()
        p.mirror.ox, p.mirror.oy = x0, y0
        _trim(p.mirror)
    _trim(p)
    return p


def _trim(p):
    """Crop a card plate to its non-empty content."""
    m = np.abs(p.rgba).max(-1) > 1e-4
    rows = np.flatnonzero(m.any(1))
    if rows.size == 0:
        p.rgba = p.rgba[:1, :1] * 0
        p.depth = p.depth[:1, :1]
        p.empty = True
        return
    cols = np.flatnonzero(m[rows[0]:rows[-1] + 1].any(0))
    r0, r1, c0, c1 = max(rows[0] - 2, 0), rows[-1] + 3, max(cols[0] - 2, 0), cols[-1] + 3
    p.rgba = np.ascontiguousarray(p.rgba[r0:r1, c0:c1])
    p.depth = np.ascontiguousarray(p.depth[r0:r1, c0:c1])
    p.ox += int(c0)
    p.oy += int(r0)
    p.empty = False
    a = p.rgba[..., 3]
    d = p.depth[a > 0.01] if (a > 0.01).any() else p.depth
    p.zrange = (float(d.min()), float(d.max()))


def warp_card(p, cam, W, H, dX, dY, dZ):
    """-> (x0, y0, block (h, w, 4) premult) in frame pixels, or None."""
    if getattr(p, 'empty', False):
        return None
    h, w = p.depth.shape
    z0, z1 = p.zrange
    xs_ = np.array([p.ox, p.ox + w, p.ox, p.ox + w], np.float64)
    ys_ = np.array([p.oy, p.oy, p.oy + h, p.oy + h], np.float64)
    fx, fy = [], []
    for z in (max(z0, dZ + 0.2), max(z1, dZ + 0.2)):
        X = (xs_ - cam.pcx) * z / cam.f
        Y = cam.h - (ys_ - cam.pcy) * z / cam.f
        zc = z - dZ
        fx.append(cam.cx + cam.f * (X - dX) / zc)
        fy.append(cam.cy - cam.f * (Y - cam.h - dY) / zc)
    fx, fy = np.concatenate(fx), np.concatenate(fy)
    bx0, bx1 = max(int(np.floor(fx.min())) - 3, 0), min(int(np.ceil(fx.max())) + 3, W)
    by0, by1 = max(int(np.floor(fy.min())) - 3, 0), min(int(np.ceil(fy.max())) + 3, H)
    if bx1 <= bx0 or by1 <= by0:
        return None
    # maps on a coarse grid (depth is smooth on a single-surface card), upsampled bilinearly
    q = 4
    bw, bh = bx1 - bx0, by1 - by0
    gw, gh = max(bw // q, 2), max(bh // q, 2)
    gx = bx0 + (np.arange(gw, dtype=np.float32) + 0.5) * (bw / gw) - 0.5
    gy = by0 + (np.arange(gh, dtype=np.float32) + 0.5) * (bh / gh) - 0.5
    xs, ys = np.meshgrid(gx, gy)
    u = xs - cam.cx
    v = ys - cam.cy
    ox = np.float32(p.ox)
    oy = np.float32(p.oy)
    z = cv2.remap(p.depth, xs + cam.mx - ox, ys + cam.my - oy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    z = np.maximum(z, dZ + 0.2)
    for _ in range(2):
        k = (z - dZ) / z
        sx = u * k + cam.f * dX / z + cam.pcx - ox
        sy = v * k - cam.f * dY / z + cam.pcy - oy
        z = np.maximum(cv2.remap(p.depth, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE), dZ + 0.2)
    k = (z - dZ) / z
    sx = cv2.resize((u * k + cam.f * dX / z + cam.pcx - ox).astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
    sy = cv2.resize((v * k - cam.f * dY / z + cam.pcy - oy).astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
    blk = cv2.remap(p.rgba, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # bicubic keeps glyph edges crisp during the slow scale-up; clamp its small ringing
    blk[..., 3] = np.clip(blk[..., 3], 0, 1)
    blk[..., :3] = np.maximum(blk[..., :3], 0)
    return bx0, by0, np.ascontiguousarray(blk)


def over_card(img, p, cam, dX, dY, dZ):
    H, W = img.shape[:2]
    r = warp_card(p, cam, W, H, dX, dY, dZ)
    if r is None:
        return
    x0, y0, blk = r
    h, w = blk.shape[:2]
    sub = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
    R.over_pm(sub, blk)
    img[y0:y0 + h, x0:x0 + w] = sub


def merge_mirrors(cards, cam, name='card_mirrors'):
    """All card reflections in one full-size plate (reflections are smeared by the wet road anyway, so they do
    not need per-card parallax precision). Cards are given far -> near."""
    rgba = np.zeros((cam.PH, cam.PW, 4), np.float32)
    zsum = np.zeros((cam.PH, cam.PW), np.float32)
    wsum = np.zeros((cam.PH, cam.PW), np.float32)
    for c in cards:
        m = c.mirror
        if m is None or getattr(m, 'empty', False):
            continue
        h, w = m.depth.shape
        sl = (slice(m.oy, m.oy + h), slice(m.ox, m.ox + w))
        a = np.clip(m.rgba[..., 3:4], 0, 1)
        rgba[sl] = rgba[sl] * (1 - a) + m.rgba
        zsum[sl] = zsum[sl] * (1 - a[..., 0]) + m.depth * a[..., 0]
        wsum[sl] = wsum[sl] * (1 - a[..., 0]) + a[..., 0]
    zr = zsum / np.maximum(wsum, 1e-6)
    depth = R.pushpull_fill(zr, np.clip(wsum * 4, 0, 1) ** 2)
    depth = cv2.GaussianBlur(depth, (0, 0), 2.0).astype(np.float32)
    return R.Plate(cam, rgba[..., :3], rgba[..., 3], depth, name)
