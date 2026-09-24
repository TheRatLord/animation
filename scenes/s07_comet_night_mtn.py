"""Round-5 landscape painting for s07_comet_night.

paint_range5  - a snow range painted as a few big light/shadow planes: curved spurs run down from the
                summits (lit face toward the comet, shadow face away, crisp edge along each spur crest,
                soft gully creases between them), snow set by ALTITUDE (a varying snowline, deep snow on
                the high peaks, bare rock ribs along spur crests, snow tongues down the couloirs, rock
                outcrops on the steep upper faces), forest below a ragged treeline that keeps the terrain
                lighting, a comet-facing crest rim that is hottest near the comet and vanishes on crests
                turned away, and aerial perspective (distant ranges bluer, paler, flatter).
forest_hills  - forested foothills as overlapping painted silhouettes (serrated conifer tops, lit
                shoulder toward the comet, body darkening downward, mist pooling between the rows) with
                stamped trees only along each row's edge - never a uniform noise field.
night_cloud_pal - clouds2 palette for moon/comet-lit night stratocumulus.
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


def _spurs(top1, Wp, y_base, rng, spacing, len_frac, lean, wob, start_jit=0.0, nseg=14):
    """Curved spur polylines running down from the crest. Returns list of (N,2) float arrays and the
    per-spur length."""
    ts = _sm1(top1, 0.01 * Wp)
    g = np.clip(np.gradient(ts), -2.5, 2.5)
    out = []
    x = rng.uniform(0, spacing)
    while x < Wp + spacing:
        xi = int(np.clip(x, 0, Wp - 1))
        depth = max(y_base - top1[xi], 1.0)
        L = depth * rng.uniform(*len_frac)
        px, py = x, top1[xi] + start_jit * depth * rng.uniform(0, 1)
        pts = [(px, py - 2)]
        drift = rng.normal(0, wob)
        curv = rng.normal(0, wob * 0.6)
        for k in range(nseg):
            q = k / nseg
            dxdy = lean * g[xi] * (1 - 0.55 * q) + drift + curv * q
            st = L / nseg
            px += dxdy * st
            py += st
            pts.append((px, py))
        out.append(np.array(pts, np.float32))
        x += spacing * rng.uniform(0.55, 1.45)
    return out


def _dist_to(lines, Wp, Hl, width=1):
    m = np.full((Hl, Wp), 255, np.uint8)
    for pl in lines:
        cv2.polylines(m, [np.round(pl).astype(np.int32)], False, 0, width, cv2.LINE_8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)


def paint_range5(Wp, Hl, top, ss, s, seed, light, pal, y_base, H, spur_px, rib_px, snow_frac=(0.35, 0.8),
                 snow_min=0.0, relief=1.0, aerial=0.0, aerial_col=None, haze_px=0.1, haze_amt=0.5,
                 rim_col=(0.7, 0.92, 1.0), rim_amt=1.0, rim_px=1.8, rim_reach=0.6, tree_line=None,
                 lean=1.0, rock_amt=1.0, steps=0.7, lit_pct=60, wob=0.18):
    """top: supersampled crest line (len Wp*ss, plate px). light: (x, y) comet position (plate px).
    pal keys: rock_sh rock_lit snow_sh snow_lit snow_hi forest_sh forest_lit haze.
    spur_px / rib_px: spacing of the major spurs / minor ribs. snow_frac: range (fraction of each peak's
    height above the valley) the snow reaches down. tree_line: None or fraction of local height above
    y_base under which forest grows. Returns RGBA plate + forest mask."""
    rng = np.random.default_rng(seed)
    pal = {k: np.asarray(v, np.float32) for k, v in pal.items()}
    ysl = np.arange(Hl, dtype=np.float32)[:, None]
    msk = np.clip(ysl - top[None, :] + 0.5, 0, 1)
    msk = cv2.resize(msk, (Wp, Hl), interpolation=cv2.INTER_AREA)
    top1 = top.reshape(Wp, ss).mean(1)
    xs, ys = C.grid(Wp, Hl)
    xr = np.arange(Wp, dtype=np.float32)
    d = np.clip(ys - top1[None, :], 0, None)
    inside = msk > 0.5

    # ---------------- structure: major spurs + minor ribs as a height field (tents around the lines)
    maj = _spurs(top1, Wp, y_base, rng, spur_px, (0.45, 1.05), lean, wob)
    mnr = _spurs(top1, Wp, y_base, rng, rib_px, (0.12, 0.45), lean, wob * 1.7, start_jit=0.12)
    # minor ribs also branch off the major spurs
    for sp in maj:
        for k in range(2, len(sp) - 3, 3):
            if rng.random() < 0.55:
                p0 = sp[k]
                side = rng.choice([-1, 1])
                L = rib_px * rng.uniform(0.8, 2.0)
                ang = math.radians(rng.uniform(25, 50)) * side
                pts = [p0]
                for j in range(1, 6):
                    q = j / 5
                    pts.append((p0[0] + math.sin(ang) * L * q, p0[1] + math.cos(ang) * L * q))
                mnr.append(np.array(pts, np.float32))
    wn = (C.fbm(max(Wp // 4, 8), max(Hl // 4, 8), 9, 3, seed=seed + 3) - 0.5)
    wn = cv2.resize(wn, (Wp, Hl), interpolation=cv2.INTER_CUBIC)
    d1 = _dist_to(maj, Wp, Hl) + wn * spur_px * 0.18
    d2 = _dist_to(mnr, Wp, Hl) + wn * rib_px * 0.25
    w1 = spur_px * 0.62
    w2 = rib_px * 0.55
    t1 = np.clip(1 - d1 / w1, 0, 1)
    t2 = np.clip(1 - d2 / w2, 0, 1)
    ramp = _ss(0.0, 0.05 * H, d)
    # crest wiggles are small local peaks whose relief dies out down the face (otherwise every jag of the
    # skyline would print a vertical streak all the way down)
    tsm = _sm1(top1, 0.012 * Wp)
    hgt0 = 0.3 * np.clip(ys - tsm[None, :], 0, None) + 0.3 * (tsm - top1)[None, :] * np.exp(-d / (0.015 * H))
    hgt = hgt0 + (t1 ** 1.1 * w1 * 0.95 + t2 ** 1.2 * w2 * 0.4 * _ss(0.02 * H, 0.06 * H, d)) * ramp * relief
    hgt = cv2.GaussianBlur(hgt.astype(np.float32), (0, 0), 0.6 * s + 0.4)
    hx = np.gradient(hgt, axis=1)
    hy = np.gradient(hgt, axis=0)
    nz = 1.0 / np.sqrt(hx * hx + hy * hy + 1)
    nx, ny = -hx * nz, -hy * nz
    lx, ly = light[0] - xs, light[1] - ys
    ln = np.sqrt(lx * lx + ly * ly) + 1e-3
    lx, ly = lx / ln, ly / ln
    # right under the comet the light is nearly vertical and would leave spur flanks equally lit: bias it
    # toward a consistent side (the Milky Way side, right) so every spur keeps one lit and one shadow plane
    lxe = lx + 0.75 * (1 - np.abs(lx))
    dot = nx * lxe * 0.95 + ny * ly * 0.55 + nz * 0.1
    # the comet is behind the range: most of the faces we see are in shade, only faces turned toward it lit
    med = float(np.percentile(dot[inside], lit_pct)) if inside.any() else 0.0
    v = np.clip(0.5 + (dot - med) * 2.6, 0, 1)
    # painted: broad flat value steps with narrow soft transitions (plus a little of the continuous ramp)
    vq = (_ss(0.36, 0.44, v) + _ss(0.62, 0.7, v)) * 0.5
    vp = steps * vq + (1 - steps) * v
    prox = np.exp(-np.abs(xs - light[0]) / (0.55 * Wp)) * np.exp(-np.abs(ys - light[1]) / (1.5 * Wp))

    # ---------------- snow by altitude
    alt = (y_base - ys)                                         # px above the valley floor
    pk = _sm1(y_base - top1, 0.03 * Wp)                         # smoothed peak height per column
    fr = snow_frac[0] + (snow_frac[1] - snow_frac[0]) * (P.fbm1d(xr / Wp, 3.5, 3, seed + 51) * 0.5 + 0.5)
    # high summits are snowed further down than low shoulders
    hrel = pk / (pk.max() + 1e-3)
    fr = fr * (0.55 + 0.6 * hrel)
    sline = y_base - top1 - pk * np.clip(fr, 0.05, 0.95)       # altitude of the snowline per column
    # neighbouring couloirs hold snow to very different depths
    sline = sline + pk * 0.28 * P.fbm1d(xr / Wp, Wp / spur_px * 0.9, 2, seed + 57)
    ts = _sm1(top1, 0.012 * Wp)
    gsl = np.clip(np.gradient(ts), -2.5, 2.5)
    xc = (xs - lean * gsl[None, :] * d).astype(np.float32)
    pad = int(0.3 * Wp)
    Tw = Wp + 2 * pad

    def tex(stretch, cells, octv, sd, warp, ang=90.0):
        t_ = P.rot_fbm(Tw, Hl, ang, stretch, Tw / cells, octv, sd, warp_amt=warp)
        return cv2.remap(t_, xc + pad, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    nA = tex(3.5, rib_px * 0.8, 4, seed + 7, 0.08)            # fall-line streaks (couloirs / rock bands)
    nB = tex(1.3, rib_px * 0.25, 3, seed + 13, 0.05)          # fine breakup
    nC = tex(6.0, rib_px * 0.9, 3, seed + 71, 0.06, ang=6.0)   # near-horizontal ledges / strata
    # 1 down the middle of each couloir, falling linearly to 0 on the spur crests -> V-shaped snow tongues
    g1 = np.clip(d1 / (0.5 * spur_px), 0, 1)
    g2 = np.clip(d2 / (0.5 * rib_px), 0, 1)
    gully = g1 ** 1.4 * (0.55 + 0.45 * g2)
    nD = tex(1.6, rib_px * 0.5, 3, seed + 19, 0.12)          # blocky patches (snow-edge lobes, outcrops)
    sn_edge = (alt - sline[None, :]) + (gully - 0.5) * 0.34 * pk[None, :] + (nD - 0.5) * 0.1 * pk[None, :] \
        + (nB - 0.5) * 0.015 * pk[None, :]
    snow = _ss(-1.5, 1.5, sn_edge) * _ss(snow_min * H - 0.01 * H, snow_min * H + 0.01 * H, alt + (gully - 0.5) * 0.02 * H)
    # bare rock: the steep shadowed flank of each spur (snow does not hold there), thin ribs along the
    # spur crests, outcrops on the steep upper faces, a few broken strata ledges
    below = _ss(0.006 * H, 0.03 * H, d + (nD - 0.5) * 0.02 * H)
    flank = _ss(0.5, 0.7, t1 + (nD - 0.5) * 0.4 + (nB - 0.5) * 0.1) * _ss(0.45, 0.2, vp) * below * \
        _ss(0.8, 0.3, d / (y_base - top1 + 1)[None, :])
    # rock ribs only along the major spurs, broken up (no comb of identical streaks under the crest)
    ribr = _ss(0.82, 0.94, t1 + (nA - 0.5) * 0.3) * _ss(0.015 * H, 0.05 * H, d) * _ss(0.35, 0.6, nB * 0.5 + nA * 0.5)
    outc = _ss(0.64, 0.72, nD * 0.75 + nB * 0.25) * _ss(0.5, 0.2, vp) * _ss(0.02 * H, 0.05 * H, d)
    strata = _ss(0.7, 0.76, nC) * _ss(0.55, 0.3, gully) * _ss(0.02 * H, 0.05 * H, d) * _ss(0.6, 0.3, vp)
    rock = np.maximum(np.maximum(flank, ribr * 0.9), np.maximum(outc * 0.85, strata * 0.7))
    rock = np.clip(rock * rock_amt, 0, 1)
    snow = snow * (1 - rock)
    # firm but painted edge of the snow patches
    snow = np.clip(snow, 0, 1)

    # ---------------- colours
    rockc = pal['rock_sh'] + (pal['rock_lit'] - pal['rock_sh']) * vp[..., None]
    rockc = rockc * (1 + 0.08 * (nB - 0.5)[..., None])
    vs = np.clip(0.18 + 0.82 * vp, 0, 1)
    snowc = pal['snow_sh'] + (pal['snow_lit'] - pal['snow_sh']) * vs[..., None]
    hi = _ss(0.72, 0.95, v) * prox
    snowc = snowc + (pal['snow_hi'] - pal['snow_lit']) * hi[..., None]
    # snow darkens slightly into the couloirs and down the face, soft wind-sculpted variation
    snowc = snowc * (1 - 0.1 * _ss(0.3, 1.0, gully)[..., None] * (1 - vp[..., None]))
    snowc = snowc * (1 + 0.05 * (nB - 0.5)[..., None])
    lk = 0.82 + 0.3 * prox[..., None]
    img = (rockc * (1 - snow[..., None]) + snowc * snow[..., None]) * lk

    forest = np.zeros((Hl, Wp), np.float32)
    if tree_line is not None:
        tl = (y_base - top1) * tree_line * (0.75 + 0.5 * (P.fbm1d(xr / Wp, 9, 3, seed + 61) * 0.5 + 0.5))
        # trees climb higher up the sheltered gullies, stop lower on the spur crests
        tb = (y_base - tl)[None, :] - (gully - 0.5) * 0.03 * H + (nB - 0.5) * 0.012 * H
        forest = _ss(-1.0, 1.0, ys - tb) * inside
        fcol = pal['forest_sh'] + (pal['forest_lit'] - pal['forest_sh']) * vp[..., None]
        fcol = fcol * (1 + 0.06 * (nB - 0.5)[..., None])
        img = img * (1 - forest[..., None]) + fcol * forest[..., None]
        snow = snow * (1 - forest)

    # ---------------- aerial perspective: whole range toward the haze colour + haze pooling at the base
    if aerial:
        ac = pal['haze'] if aerial_col is None else np.asarray(aerial_col, np.float32)
        img = img * (1 - aerial) + ac * aerial
    hz = _ss(y_base - haze_px, y_base + 0.1 * haze_px, ys) * haze_amt
    img = img * (1 - hz[..., None]) + pal['haze'] * hz[..., None]

    # ---------------- crest rim toward the comet (only where the silhouette faces it)
    m8 = inside.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3).astype(np.float32)
    gm = C.blur(msk, 2.0 * s + 0.5)
    gx, gy = np.gradient(gm, axis=1), np.gradient(gm, axis=0)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    facing = np.clip((-gx * lx - gy * ly) / gn, 0, 1) ** 1.5
    near = np.exp(-np.abs(xs - light[0]) / (rim_reach * Wp))
    rw = (rim_px * s + 0.3) * (0.6 + 0.8 * near)
    rim = (np.exp(-dist / rw) * 0.8 + np.exp(-dist / (rw * 4)) * 0.2) * facing * near
    rim *= (0.35 + 0.65 * snow) * (1 - forest)
    img = img + rim[..., None] * np.asarray(rim_col, np.float32) * rim_amt
    return np.dstack([img, msk]).astype(np.float32), forest.astype(np.float32), snow.astype(np.float32)


def forest_hills(Wp, Hl, rows, ss, s, H, light, seed, y_base, tree_h, cols, mist_col, mist_amt=0.35,
                 rim_col=(0.1, 0.2, 0.36), edge_trees=1.0, stamp=False, dens=None, back_haze=0.35):
    """Forested foothills as overlapping painted rows, back to front.
    rows: list of supersampled top lines (len Wp*ss), back first. cols: dict top_lit top_sh base.
    Each row: serrated conifer skyline (tree bumps), lit shoulder toward the comet (slope of the row's
    top line), body darkening downward, mist pooling at its foot (so the next row's crisp top reads).
    Trees are stamped only in a band along each row's top edge. Returns straight RGBA."""
    rng = np.random.default_rng(seed)
    out = np.zeros((Hl, Wp, 3), np.float32)
    A = np.zeros((Hl, Wp), np.float32)
    ysl = np.arange(Hl, dtype=np.float32)[:, None]
    xs2 = np.arange(Wp * ss, dtype=np.float32) / ss
    xr = np.arange(Wp, dtype=np.float32)
    ct, cs, cb = [np.asarray(cols[k], np.float32) for k in ('top_lit', 'top_sh', 'base')]
    nrow = len(rows)
    for i, top in enumerate(rows):
        f = i / max(nrow - 1, 1)                       # 0 back .. 1 front
        th = tree_h * (0.75 + 0.5 * f)
        bumps = P.tree_bumps(xs2, 0, Wp, th * 0.34, th, seed + 17 * i, jitter=0.6, width_ratio=0.3, dens=dens)
        tt = top - bumps
        m = np.clip(ysl - tt[None, :] + 0.5, 0, 1)
        m = cv2.resize(m, (Wp, Hl), interpolation=cv2.INTER_AREA)
        t1 = top.reshape(Wp, ss).mean(1)
        ts = _sm1(t1, 0.01 * Wp)
        g = np.gradient(ts)                             # >0: row descends to the right
        side = np.sign(light[0] - xr)                   # comet to the right -> right-facing slopes lit
        lit = np.clip(0.5 + 1.6 * (-g * side) + 0.0, 0, 1)
        lit = _sm1(lit, 0.004 * Wp)
        near = np.exp(-np.abs(xr - light[0]) / (0.6 * Wp))
        dd = np.clip(ysl - t1[None, :], 0, None)
        topc = cs + (ct - cs) * (lit * (0.5 + 0.5 * near))[:, None]
        k = _ss(0, 0.07 * H, dd)[..., None]
        col = topc[None, :, :] * (1 - k) + cb * k
        # broad painted value variation (a few soft patches), not texture
        pn = P.fbm_lowres(Wp, Hl, 4, 2, seed + 5 + i, q=8)
        col = col * (1 + 0.1 * (pn - 0.5)[..., None])
        # faint vertical crown texture (rows of conifers), strongest just under the serrated top
        if i == 0:
            vt = P.rot_fbm(Wp, Hl, 90.0, 4.0, Wp / (th * 0.45), 2, seed + 91)
        col = col * (1 + 0.07 * (vt - 0.5)[..., None] * (0.4 + 0.6 * np.exp(-dd / (th * 2)))[..., None])
        # distance haze: back rows paler
        col = col * (1 - back_haze * (1 - f)) + mist_col * back_haze * (1 - f)
        rgba = np.dstack([col, m]).astype(np.float32)
        if stamp:
            band = m * (1 - _ss(th * edge_trees, th * edge_trees * 1.6, dd))
            V.stamp_forest(rgba, band, seed + 31 * i, s, light[0], lambda y: th * 0.95,
                           cb * 0.7 + cs * 0.3, cs * 0.8 + ct * 0.2, spacing=1.05, rim_col=None)
        # crisp tree-top rim toward the comet
        sh = np.zeros_like(m)
        dpx = max(int(round(1.1 * s)), 1)
        sh[dpx:] = m[:-dpx]
        rim = np.clip(m - sh, 0, 1) * (0.25 + 0.75 * near)[None, :] * (0.4 + 0.6 * lit)[None, :]
        rgba[..., :3] += rim[..., None] * np.asarray(rim_col, np.float32)
        a = rgba[..., 3]
        out = out * (1 - a[..., None]) + rgba[..., :3] * a[..., None]
        A = A + a * (1 - A)
        # mist pooling at the foot of this row (lightens what is behind the next row)
        if i < nrow - 1:
            nxt = rows[i + 1].reshape(Wp, ss).mean(1)
            mk = _ss(nxt[None, :] - 0.05 * H, nxt[None, :] + 0.004 * H, ysl) * mist_amt * A
            out = out * (1 - mk[..., None]) + mist_col * mk[..., None]
    # final mist toward the waterline
    mk = _ss(y_base - 0.04 * H, y_base, ysl) * mist_amt * 0.9 * A
    out = out * (1 - mk[..., None]) + mist_col * mk[..., None]
    rgb = out / np.maximum(A, 1e-5)[..., None]
    return np.dstack([rgb, A]).astype(np.float32)


def night_cloud_pal():
    """clouds2 palette for low stratocumulus lit by the comet at night (cool, dim, cyan-silver rims)."""
    return dict(hi=(0.5, 0.66, 0.9), lit=(0.4, 0.55, 0.82), lit_lo='#4f6cab', mid='#415b99',
                shade='#2f4786', deep='#233873', refl='#36528f', bounce='#3a4c8a', rim=(0.7, 0.95, 1.15),
                haze='#3a62a6', edge_dark=0.02)
