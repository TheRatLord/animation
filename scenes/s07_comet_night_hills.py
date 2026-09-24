"""Round-7 foothills / lake-light helpers for s07_comet_night.

forest_hills2 - forested foothills painted as snowy slopes with real form: every row is a small terrain
                (spurs and gullies running down the fall line) lit by the comet -> moonlit upper faces,
                cool shadowed gullies. The cover is a clumped forest (irregular groves elongated down the
                fall line, denser low and in the gullies) with snow clearings and scattered snow patches
                between them; the conifers are stamped one by one so each grove has a serrated edge
                against the snow. The skyline is clumped (varying canopy heights, gaps, clearings where
                the snowy ridge shows), with a soft comet-lit rim on the top edge. Back rows fade into the
                haze colour.
light_streaks - lake reflections of the town lights with varied length / width, ripple wobble and
                broken dashes (nearest lights -> longest streaks).
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P
import s07_comet_night_v3 as V


def _sm1(a, sigma):
    return cv2.GaussianBlur(np.asarray(a, np.float32).reshape(1, -1), (0, 0), sigmaX=max(sigma, 0.5),
                            borderType=cv2.BORDER_REFLECT)[0]


def _ss(e0, e1, x):
    return C.smoothstep(e0, e1, x)


def clumped_bumps(xs, Wp, th, seed, gap=-0.5, dens=None):
    """Conifer skyline: trees grouped in clumps of varying height, with gaps / clearings."""
    rng = np.random.default_rng(seed)
    xf = xs / Wp
    cl = P.fbm1d(xf, 22.0, 3, seed + 3)                     # clump field (-1..1)
    hm = 0.45 + 0.85 * _ss(gap, 0.55, cl)                    # canopy height multiplier
    out = np.zeros_like(xs, dtype=np.float32)
    x = float(xs[0])
    x1 = float(xs[-1])
    n = len(xs)
    while x < x1:
        i = min(int(np.searchsorted(xs, x)), n - 1)
        sp = th * 0.3 * rng.uniform(0.6, 1.4)
        if cl[i] < gap + 0.12 * rng.normal() or (dens is not None and rng.random() > dens(x)):
            x += sp
            continue
        hgt = th * hm[i] * rng.uniform(0.7, 1.25)
        w = hgt * 0.3 * rng.uniform(0.8, 1.2)
        i0 = np.searchsorted(xs, x - w)
        i1 = np.searchsorted(xs, x + w)
        seg = xs[i0:i1]
        prof = hgt * np.clip(1 - np.abs(seg - x) / w, 0, 1) ** 1.15
        out[i0:i1] = np.maximum(out[i0:i1], prof)
        x += sp
    return out, cl


def forest_hills2(Wp, Hl, rows, ss, s, H, light, seed, y_base, tree_h, pal, mist_col, mist_amt=0.3,
                  back_haze=0.45, rim_col=(0.45, 0.7, 0.95), rim_amt=1.0, dens=None, forest_bias=0.0,
                  spur_px=None, lean=1.0):
    """rows: supersampled top lines (len Wp*ss), back first.
    pal: snow_lit snow_sh snow_gully forest_lit forest_sh forest_base.  Returns straight RGBA."""
    pal = {k: np.asarray(v, np.float32) for k, v in pal.items()}
    mist_col = np.asarray(mist_col, np.float32)
    out = np.zeros((Hl, Wp, 3), np.float32)
    A = np.zeros((Hl, Wp), np.float32)
    ysl = np.arange(Hl, dtype=np.float32)[:, None]
    xs2 = np.arange(Wp * ss, dtype=np.float32) / ss
    xr = np.arange(Wp, dtype=np.float32)
    xs, ys = C.grid(Wp, Hl)
    nrow = len(rows)
    lxw = float(light[0])
    spur_px = spur_px or 0.045 * Wp
    for i, top in enumerate(rows):
        f = i / max(nrow - 1, 1)                       # 0 back .. 1 front
        th = tree_h * (0.7 + 0.55 * f)
        bumps, _ = clumped_bumps(xs2, Wp, th, seed + 17 * i, dens=dens)
        tt = top - bumps
        m = np.clip(ysl - tt[None, :] + 0.5, 0, 1)
        m = cv2.resize(m, (Wp, Hl), interpolation=cv2.INTER_AREA)
        t1 = top.reshape(Wp, ss).mean(1)
        # depth of this row at each column (to the next row's top, or the waterline)
        nxt = rows[i + 1].reshape(Wp, ss).mean(1) if i < nrow - 1 else np.full(Wp, y_base, np.float32)
        depth = np.maximum(nxt - t1, 2.0)
        dd = np.clip(ys - t1[None, :], 0, None)
        q = np.clip(dd / depth[None, :], 0, 1.5)       # 0 at the ridge .. 1 at the next row
        ts = _sm1(t1, 0.01 * Wp)
        g = np.clip(np.gradient(ts), -2.0, 2.0)
        # ---- form: spurs / gullies following the fall line (lean with the local slope of the ridge)
        xc = xs - lean * g[None, :] * dd * 1.6
        sp_ = spur_px * (0.8 + 0.4 * f)
        R = P.fbm1d(xc / Wp, Wp / sp_, 3, seed + 5 + 7 * i, gain=0.45)          # -1..1 ridge/gully
        R = R + 0.35 * P.fbm1d(xc / Wp + 0.013 * dd / depth[None, :], Wp / (sp_ * 0.4), 2, seed + 9 + 7 * i)
        amp = _ss(0.0, 0.25, q) * np.minimum(depth[None, :], 0.12 * H) * 0.55
        hgt = -dd * 0.6 + R * amp
        hgt = cv2.GaussianBlur(hgt.astype(np.float32), (0, 0), 1.2 * s + 0.4)
        hx = np.gradient(hgt, axis=1)
        side = np.clip((lxw - xs) / (0.25 * Wp), -1, 1)
        side = np.where(np.abs(side) < 0.35, 0.35 * np.sign(side + 1e-3), side)   # consistent lit side
        dot = hx * side * 1.8 + 0.25
        v = np.clip(0.5 + dot, 0, 1)
        v = 0.65 * (_ss(0.38, 0.48, v) * 0.5 + _ss(0.62, 0.72, v) * 0.5) + 0.35 * v   # painted steps
        # moonlit upper faces: the slope just under each ridge catches the most light
        v = np.clip(v + 0.2 * np.exp(-q / 0.18) - 0.15 * _ss(0.4, 1.0, q), 0, 1)
        gully = _ss(0.05, -0.55, R)                    # 1 in the gully floors
        near = np.exp(-np.abs(xs - lxw) / (0.5 * Wp))
        # ---- cover: clumped groves elongated down the fall line + snow clearings / patches
        pad = int(0.3 * Wp)
        Tw = Wp + 2 * pad
        nb = P.rot_fbm(Tw, Hl, 90.0, 3.5, Tw / (th * 7.0), 4, seed + 41 + i, warp_amt=0.05)
        nb = cv2.remap(nb, (xc + pad).astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        nf = P.rot_fbm(Tw, Hl, 80.0, 1.4, Tw / (th * 2.2), 3, seed + 61 + i)
        nf = cv2.remap(nf, (xc + pad).astype(np.float32), ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cb = 0.3 * P.fbm1d(xc / Wp, 9.0, 2, seed + 77 + i)             # whole slopes more / less wooded
        fd = (nb - 0.5) * 1.9 + 0.35 * gully + 0.3 * _ss(0.2, 0.9, q) + 0.12 * np.exp(-q / 0.08) \
            + 0.2 * (nf - 0.5) + cb + forest_bias + 0.14
        if dens is not None:
            dm = np.array([dens(x) for x in xr], np.float32)
            fd = fd - 1.5 * (1 - dm)[None, :]
        fmask = _ss(-0.04, 0.04, fd) * m
        # ---- colours
        vs = np.clip(0.1 + 0.9 * v, 0, 1)[..., None]
        snow = pal['snow_sh'] + (pal['snow_lit'] - pal['snow_sh']) * vs
        snow = snow * (1 - 0.3 * gully[..., None] * (1 - vs)) + pal['snow_gully'] * 0.0
        snow = snow * (0.88 + 0.22 * near[..., None]) * (1 - 0.22 * f) * (1 - 0.18 * _ss(0.3, 1.0, q)[..., None])
        snow = snow + (pal['snow_lit'] * 0.15) * (np.exp(-q / 0.1) * near)[..., None]
        fcol = pal['forest_sh'] + (pal['forest_lit'] - pal['forest_sh']) * vs
        fk = _ss(0.0, 1.0, q)[..., None]
        fcol = fcol * (1 - 0.5 * fk) + pal['forest_base'] * 0.5 * fk
        body = snow * (1 - fmask[..., None]) + fcol * fmask[..., None]
        # the gap between rows sinks into cool shade low down (valley floor in shadow)
        body = body * (1 - 0.25 * _ss(0.6, 1.1, q)[..., None])
        rgba = np.dstack([body, m]).astype(np.float32)
        # ---- conifers stamped over the groves: serrated grove edges against the snow
        reg = (fmask > 0.5).astype(np.float32)
        if reg.any():
            cd = pal['forest_base'] * 0.55 + pal['forest_sh'] * 0.45
            cl_ = pal['forest_sh'] * 0.4 + pal['forest_lit'] * 0.6
            szf = lambda y: th * 0.8  # noqa: E731
            before = rgba[..., :3].copy()
            V.stamp_forest(rgba, reg, seed + 131 * i + 7, s, lxw, szf, cd, cl_, spacing=1.15, ss=2)
            # the canopy keeps the terrain form: lit spur faces lighter, gullies / low slopes darker,
            # and big soft value patches across each grove (never one flat stamped tone)
            pn = P.fbm_lowres(Wp, Hl, 6, 2, seed + 55 + i, q=8)
            shade = (0.5 + 0.65 * vs[..., 0]) * (1 - 0.3 * gully) * (1 - 0.3 * _ss(0.3, 1.0, q)) \
                * (0.85 + 0.3 * pn) * (0.8 + 0.3 * near)
            tree = np.clip(np.abs(rgba[..., :3] - before).sum(-1) * 40, 0, 1)
            rgba[..., :3] = rgba[..., :3] * (1 + (shade[..., None] - 1) * np.maximum(tree, fmask)[..., None])
        # ---- aerial: back rows paler / bluer
        hz = back_haze * (1 - f)
        rgba[..., :3] = rgba[..., :3] * (1 - hz) + mist_col * hz
        # ---- soft comet-lit rim along the top edge (snowy ridge brighter than tree tips)
        a = rgba[..., 3]
        sh1 = np.zeros_like(a)
        d1 = max(int(round(1.2 * s)), 1)
        sh1[d1:] = a[:-d1]
        edge = np.clip(a - sh1, 0, 1)
        edge = np.maximum(edge, C.blur(edge, 1.2 * s + 0.3) * 0.8)
        gap_top = (bumps.reshape(Wp, ss).mean(1) < 0.15 * th).astype(np.float32)
        rimw = (0.25 + 0.75 * np.exp(-np.abs(xr - lxw) / (0.35 * Wp))) * (0.55 + 0.45 * gap_top)
        rimw = _sm1(rimw, 2.0)
        rgba[..., :3] += (edge * rimw[None, :] * (0.45 + 0.55 * f))[..., None] * np.asarray(rim_col, np.float32) \
            * rim_amt
        out = out * (1 - a[..., None]) + rgba[..., :3] * a[..., None]
        A = A + a * (1 - A)
        # mist pooling at the foot of this row (separates it from the next row's crisp top)
        if i < nrow - 1:
            mk = _ss(nxt[None, :] - 0.035 * H, nxt[None, :] + 0.003 * H, ysl) * mist_amt * A
            out = out * (1 - mk[..., None]) + mist_col * mk[..., None]
    mk = _ss(y_base - 0.03 * H, y_base, ysl) * mist_amt * 0.8 * A
    out = out * (1 - mk[..., None]) + mist_col * mk[..., None]
    rgb = out / np.maximum(A, 1e-5)[..., None]
    return np.dstack([rgb, A]).astype(np.float32)


def light_streaks(Hlk, Wp, lights, y_h, lk0, H, s, seed=5):
    """Reflection streaks of point lights on the lake: varied length / width, wobbling with the ripples
    and broken into dashes. Lights nearer the camera (lower on the far shore) get the longest streaks."""
    rng = np.random.default_rng(seed)
    lk = np.zeros((Hlk, Wp, 3), np.float32)
    for (x0, y0, col, it, r) in lights:
        ym = 2 * y_h - y0 - lk0
        prox = np.clip((y0 - (y_h - 0.06 * H)) / (0.06 * H), 0, 1)       # 1 = right on the waterline
        Ls = (0.026 + 0.06 * prox + 0.025 * it) * H * rng.uniform(0.45, 1.6)
        wdt = (max(min(r, 3 * s), 1.0) * 1.1 + 1.2 * s) * rng.uniform(0.7, 1.4)
        x0i, x1i = int(max(x0 - 9 * wdt - 6 * s, 0)), int(min(x0 + 9 * wdt + 6 * s + 1, Wp))
        y0i, y1i = int(max(ym - 2, 0)), int(min(ym + Ls * 3, Hlk))
        if x1i <= x0i or y1i <= y0i:
            continue
        yy2, xx2 = np.mgrid[y0i:y1i, x0i:x1i].astype(np.float32)
        dyy = yy2 - ym
        ph = rng.uniform(0, 6.28)
        fr = rng.uniform(0.18, 0.32) / (s + 0.2)
        wob = (np.sin(dyy * fr + ph) * 0.6 + np.sin(dyy * fr * 2.3 + ph * 1.7) * 0.4) * (0.6 + dyy / (0.03 * H)) * s
        wd = wdt * (1 + dyy / (0.08 * H))
        dash = 0.35 + 0.65 * _ss(-0.2, 0.5, np.sin(dyy * fr * 1.7 + ph * 2.1) + 0.5 * np.sin(dyy * fr * 3.9 + ph))
        k = np.exp(-((xx2 - x0 - wob) / wd) ** 2) * np.exp(-np.clip(dyy, 0, None) / Ls) * \
            C.smoothstep(-2, 1, dyy) * dash
        lk[y0i:y1i, x0i:x1i] += k[..., None] * np.asarray(col, np.float32) * it * 0.8
    return lk


def shore_mist(Wp, Hl, y_h, H, s, light_x, warm_x, seed=71):
    """Straight RGBA: a soft, slightly luminous mist band lying over the far shoreline - behind the town,
    in front of the far forest - so the roof silhouettes separate. Torn horizontal streaks, brightest
    under the comet, warmed by the town lights."""
    from lib import clouds as K
    ys = np.arange(Hl, dtype=np.float32)[:, None]
    xs = np.arange(Wp, dtype=np.float32)[None, :]
    st = K.streaks(Wp, Hl, seed, xcells=6.0, ycells=60.0, octaves=4)
    st2 = K.streaks(Wp, Hl, seed + 1, xcells=11.0, ycells=110.0, octaves=3)
    yc = y_h - 0.016 * H
    prof = np.where(ys < yc, np.exp(-((ys - yc) / (0.017 * H)) ** 2), np.exp(-((ys - yc) / (0.014 * H)) ** 2))
    prof = prof * _ss(y_h + 0.002 * H, y_h - 0.004 * H, ys)
    lit = np.exp(-np.abs(xs - light_x) / (0.4 * Wp))
    warm = np.exp(-((xs - warm_x) / (0.14 * Wp)) ** 2)
    a = prof * (0.3 + 0.35 * _ss(0.25, 0.85, st) + 0.15 * (st2 - 0.5)) * (0.8 + 0.35 * lit)
    # a thinner upper wisp layer drifting over the foot of the far forest
    yw = y_h - 0.036 * H
    a = a + np.exp(-((ys - yw) / (0.006 * H)) ** 2) * _ss(0.45, 0.9, st2) * 0.3 * (0.6 + 0.4 * lit)
    a = np.clip(a, 0, 0.72).astype(np.float32)
    top = np.clip((yc - ys) / (0.02 * H) + 0.5, 0, 1)
    col = (np.array([0.3, 0.44, 0.74], np.float32) + (lit * (0.5 + 0.5 * top))[..., None] *
           np.array([0.1, 0.17, 0.24], np.float32) + warm[..., None] * np.array([0.2, 0.1, 0.02], np.float32))
    col = np.broadcast_to(col, (Hl, Wp, 3))
    return np.dstack([col, a]).astype(np.float32)
