"""s05_sakura helpers: painted grass-blade tufts near camera (curb edge + grass bank).

Tufts are rasterised once into a few horizontal band plates (grouped by the screen row of their roots) and
composited per frame with the ground's per-row parallax shift taken at each band's root row, so the blades
stay glued to the ground plane while the camera dollies.
"""
import math
import numpy as np
import cv2

import s05_sakura_env as E


def _blade_poly(p0, p1, bend, w0, n=5):
    """Tapered curved blade polygon from root p0 to tip p1 (screen px), bend = lateral px at the middle."""
    d = p1 - p0
    L = np.linalg.norm(d) + 1e-6
    t = np.linspace(0, 1, n)
    nrm = np.array([-d[1], d[0]]) / L
    c = p0[None, :] + d[None, :] * t[:, None] + nrm[None, :] * (bend * t ** 1.6)[:, None]
    w = w0 * (1 - t) ** 0.9 + 0.15
    left = c + nrm[None, :] * w[:, None]
    right = c - nrm[None, :] * w[:, None]
    return np.concatenate([left, right[::-1]], 0)


def build(cam, W, H, emx, ground, seed=31, tufts=None, nb=10):
    """-> list of bands dict(pm=(h, Wc, 4) premultiplied, r0=top row, root_row=float)."""
    rng = np.random.default_rng(seed)
    Wc = W + 2 * emx
    f = cam.f
    if tufts is None:
        tufts = []
        # along the curb edge (spilling over it) and scattered on the bank, denser near camera
        for s in np.arange(1.2, 16.0, 0.09):
            if rng.random() < 0.55:
                tufts.append((E.U_PATH1 + 0.3 + rng.normal(0, 0.05), s + rng.uniform(-0.04, 0.04), 1.0))
        for k in range(260):
            s = 1.0 + 11.0 * rng.random() ** 1.6
            u = E.U_PATH1 + 0.4 + rng.uniform(0, 3.2)
            tufts.append((u, s, 0.85))
    lum = ground[..., :3].mean(-1)
    # sun direction on screen (blades lit on the side facing upper-left)
    items = []
    for (u, s, amt) in tufts:
        X, Y, Z = cam.to_cam(u, E.Y_PATH, s)
        if Z < 1.2:
            continue
        x0, y0 = cam.proj(X, Y, Z, 0.0, emx)
        if y0 < cam.hy or y0 > H + 20 or x0 < 0 or x0 > Wc:
            continue
        iy, ix = int(min(max(y0, 0), H - 1)), int(min(max(x0, 0), Wc - 1))
        lg = float(lum[max(iy - 2, 0), ix])
        items.append((Z, u, s, amt, x0, y0, lg))
    items.sort(key=lambda r: -r[0])
    rows = np.array([it[5] for it in items]) if items else np.zeros(0)
    if len(rows) == 0:
        return []
    edges = np.quantile(rows, np.linspace(0, 1, nb + 1))
    bands = []
    ss = 2
    for b in range(nb):
        sel = [it for it in items if edges[b] <= it[5] <= edges[b + 1] + (1e-6 if b == nb - 1 else 0)]
        if not sel:
            continue
        rr = np.array([it[5] for it in sel])
        root_row = float(np.median(rr))
        top = int(max(0, rr.min() - 0.3 * f / min(it[0] for it in sel)))
        bot = int(min(H, rr.max() + 4))
        if bot <= top:
            continue
        h = bot - top
        acc = np.zeros((h * ss, Wc * ss, 3), np.float32)
        al = np.zeros((h * ss, Wc * ss), np.float32)
        for (Z, u, s, amt, x0, y0, lg) in sel:
            sun = float(np.clip((lg - 0.3) / 0.35, 0, 1))
            nbl = int(rng.integers(5, 12))
            for j in range(nbl):
                hgt = rng.uniform(0.06, 0.2) * amt
                lean_u = rng.normal(-0.02, 0.05)
                lean_s = rng.normal(0, 0.04)
                du = rng.normal(0, 0.025)
                X0, Y0, Z0 = cam.to_cam(u + du, E.Y_PATH, s + rng.normal(0, 0.02))
                X1, Y1, Z1 = cam.to_cam(u + du + lean_u, E.Y_PATH + hgt, s + lean_s)
                p0 = np.array(cam.proj(X0, Y0, Z0, 0.0, emx), np.float64)
                p1 = np.array(cam.proj(X1, Y1, Z1, 0.0, emx), np.float64)
                p0[1] -= top
                p1[1] -= top
                w0 = max(0.5, f * 0.0035 / Z0)
                bend = rng.normal(0, 0.25) * np.linalg.norm(p1 - p0)
                poly = _blade_poly(p0 * ss, p1 * ss, bend * ss, w0 * ss)
                bx0 = int(max(poly[:, 0].min() - 2, 0))
                bx1 = int(min(poly[:, 0].max() + 3, Wc * ss))
                by0 = int(max(poly[:, 1].min() - 2, 0))
                by1 = int(min(poly[:, 1].max() + 3, h * ss))
                if bx1 <= bx0 or by1 <= by0:
                    continue
                sub = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                cv2.fillPoly(sub, [np.round((poly - [bx0, by0]) * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
                a = sub.astype(np.float32) / 255.0
                # colour: dark teal root -> lit yellow-green tip (sunlit) / blue-green (shade)
                yy = np.arange(by0, by1, dtype=np.float32)[:, None]
                tt = np.clip((p0[1] * ss - yy) / max(p0[1] * ss - p1[1] * ss, 1.0), 0, 1)
                tone = rng.uniform(-0.08, 0.08)
                lit_c = np.array([0.72 + tone, 0.9 + tone * 0.5, 0.36], np.float32)
                shd_c = np.array([0.18, 0.4 + tone, 0.4], np.float32)
                tip = shd_c * (1 - sun) + lit_c * sun
                root = np.array([0.04, 0.13, 0.12], np.float32) * (0.8 + 0.4 * sun)
                col = root[None, None, :] * (1 - tt[..., None]) ** 1.2 + tip[None, None, :] * (1 - (1 - tt[..., None]) ** 1.2)
                if sun > 0.5 and rng.random() < 0.3:
                    col = col + np.array([0.25, 0.22, 0.1], np.float32) * tt[..., None] ** 3   # glinting tip
                ca = acc[by0:by1, bx0:bx1]
                aa = al[by0:by1, bx0:bx1]
                ca *= (1 - a)[..., None]
                ca += col * a[..., None]
                aa *= (1 - a)
                aa += a
        pm = np.dstack([acc, al])
        pm = cv2.resize(pm, (Wc, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        # convert to straight alpha for over_shift (which multiplies colour by alpha)
        a = pm[..., 3:4]
        st = np.concatenate([pm[..., :3] / np.maximum(a, 1e-4), a], -1).astype(np.float32)
        bands.append(dict(rgba=st, r0=top, root_row=root_row))
    return bands


def composite(img, bands, emx, shift_fn):
    for b in bands:
        E.over_shift(img, b['rgba'], -emx, b['r0'], float(shift_fn(b['root_row'])))
