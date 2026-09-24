"""s01 helper (round 8): the hero cumulonimbus rebuilt with lib/clouds3's volumetric painter.

A tall towering cumulonimbus (~1.8x taller than wide): a broad hazy base hidden behind the town, a
column of stacked heads that recede into depth as they rise, a big cauliflower crown. The sun sits just
behind the crown's upper-left shoulder (backlit three-quarter light): the sun-facing planes are near
white, the body turns into a saturated cool blue-violet shadow core, and the crown silhouette near the
sun burns with a bright forward-scatter rim."""
import math
import numpy as np
import cv2

from lib import clouds3 as K3


KK = K3
F32 = np.float32
# saturated cool ramp: deep cobalt-violet core -> cornflower shade -> pale lilac turn -> near-white lit
K3.VRAMPS['s01'] = dict(stops=[(0.0, (0.27, 0.38, 0.78)), (0.2, (0.38, 0.5, 0.87)), (0.38, (0.57, 0.67, 0.93)),
                               (0.52, (0.78, 0.83, 0.96)), (0.64, (0.93, 0.93, 0.96)), (0.8, (0.985, 0.98, 0.965)),
                               (1.0, (1.03, 1.01, 0.97))], hi=(1.05, 1.03, 0.97))


def _prof_main(ph):
    def prof(s):
        # broad base -> waist -> swelling cauliflower crown -> rounded top
        base = 0.95 - 0.3 * min(s / 0.55, 1.0) ** 0.8
        crown = 0.12 * math.exp(-((s - 0.78) / 0.14) ** 2)
        wob = 0.05 * math.sin(2.6 * math.pi * s + ph)
        return max(base + crown + wob, 0.2) * math.sqrt(max(1 - s ** 8, 0.04))
    return prof


def custom_lobes(rng, X, Yb, Z, Hc, L, hw0=0.27, hw1=0.19, crown=0.26, lean=-0.04, side_n=7, front_n=10,
                 sizes=(0.07, 0.035, 0.017), dens=0.95, flat=0.9, depth=0.85, recede=0.18, hier=0.9, clump=0.6,
                 front=0.92, crown_heads=5, side_r=(0.5, 0.75), seedx=0.0, taper=1.0, turrets=(), anvil=None):
    """Hand-designed massing (camera space px at depth Z): a continuous column (no waist) with staggered
    bulging heads on both flanks, a few big heads on the camera face, and a crown cluster of big heads;
    then medium / fine cauliflower grown on the exposed upper / sunward surface."""
    ph = rng.uniform(0, 6.28, 4)

    def ax(s):
        return X + lean * Hc * s + 0.03 * Hc * math.sin(2.0 * math.pi * s + ph[0])

    def hw(s):
        w = hw0 + (hw1 - hw0) * min(s / 0.85, 1.0) ** taper + (crown - hw1) * math.exp(-((s - 0.86) / 0.1) ** 2)
        return Hc * w * math.sqrt(max(1 - (s / 1.0) ** 10, 0.05))

    cores = []
    # spine
    n = 14
    for i in range(n):
        s = (i + 0.5) / n
        r = hw(s) * 0.8
        y = Yb - s * Hc + r * flat * 0.4
        y = max(y, Yb - Hc + r * flat)
        cores.append((np.array([ax(s), y, Z + recede * s * Hc]), np.array([r * 1.05, r * flat, r * depth])))
    # staggered flank heads
    for side in (-1, 1):
        s = rng.uniform(0.02, 0.1)
        k = 0
        while s < 0.93 and k < 30:
            pw = hw(s)
            r = pw * rng.uniform(*side_r)
            x = ax(s) + side * (pw - r * rng.uniform(0.45, 0.75))
            y = Yb - s * Hc + r * flat * 0.3
            z = Z + recede * s * Hc + rng.uniform(-0.35, 0.1) * r
            cores.append((np.array([x, y, z]), np.array([r * rng.uniform(1.0, 1.15), r * flat, r * depth])))
            s += r / Hc * rng.uniform(1.1, 1.6)
            k += 1
    # big heads on the camera face (break the column into overlapping masses)
    for i in range(front_n):
        s = rng.uniform(0.05, 0.85)
        pw = hw(s)
        r = pw * rng.uniform(0.45, 0.7)
        x = ax(s) + rng.uniform(-0.5, 0.5) * pw
        y = Yb - s * Hc + r * flat * 0.2
        z = Z + recede * s * Hc - pw * rng.uniform(0.25, 0.5)
        cores.append((np.array([x, y, z]), np.array([r * rng.uniform(1.0, 1.2), r * flat, r * depth])))
    # crown cluster: one dominant dome + satellites (a turret rising on the sun side)
    for i in range(crown_heads):
        s = 0.9 if i == 0 else rng.uniform(0.74, 0.95)
        pw = hw(s)
        r = pw * (0.85 if i == 0 else rng.uniform(0.4, 0.62))
        x = ax(s) + (0 if i == 0 else rng.uniform(-0.75, 0.75) * pw)
        y = max(Yb - s * Hc + r * flat * 0.5, Yb - Hc * rng.uniform(0.97, 1.0) + r * flat)
        z = Z + recede * s * Hc - rng.uniform(0.0, 0.3) * pw
        cores.append((np.array([x, y, z]), np.array([r * 1.1, r * flat, r * depth])))
    # stepped turrets on the flanks: (x offset / Hc, top height / Hc, radius / Hc, z offset / Hc)
    for (xo, th, rr, zo) in turrets:
        s = 0.0
        xb = X + xo * Hc
        while True:
            r = rr * Hc * (1 - 0.25 * s) * rng.uniform(0.85, 1.1)
            y = Yb - s * th * Hc
            if y - r * flat < Yb - th * Hc:
                y = Yb - th * Hc + r * flat
                cores.append((np.array([xb + xo * 0.2 * s * Hc, y, Z + zo * Hc + recede * th * s * Hc]),
                              np.array([r * 1.1, r * flat, r * depth])))
                break
            cores.append((np.array([xb + xo * 0.2 * s * Hc, y, Z + zo * Hc + recede * th * s * Hc]),
                          np.array([r * 1.1, r * flat, r * depth])))
            s += r / (th * Hc) * 0.9
    lob = KK.surface_grow(rng, cores, [Hc * f for f in sizes], dens=dens, area=math.pi * hw0 * Hc * Hc * 1.3,
                          sun=L, flat=flat, clump=clump, front=front, hier=hier)
    rows = []
    for c, r, rt in lob:
        rows.append((c[0], c[1], c[2], r[0], r[1], r[2], -1e30, Yb + rng.normal() * 0.003 * Hc,
                     1.0, 0.0, 0.0, rt, 0.0))
    rows = np.asarray(rows, np.float64)
    if anvil:
        a = dict(anvil)
        top = Yb - Hc * a.get('top', 1.0)
        an = K3.anvil_lobes(rng, ax(1.0) + a.get('dx', 0.0) * Hc, top, Z + recede * Hc, a['left'] * Hc,
                            a['right'] * Hc, a['thick'] * Hc, a.get('col_hw', hw1) * Hc, neck=a.get('neck', 2.2),
                            sun3=L)
        rows = np.concatenate([rows, an], 0)
    return rows


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _lerp3(a, b, t):
    return np.asarray(a, F32) + (np.asarray(b, F32) - np.asarray(a, F32)) * t[..., None]


PAINT = dict(k_sun=0.7, k_amb=0.3, tone_k=0.5, gamma=0.8, lam_floor=0.15, lam_wrap=0.3,
             term=0.5, term_e=0.025, warp=0.05, pre_kuwa=5, post_kuwa=3,
             deep=(0.28, 0.37, 0.78), shade=(0.43, 0.53, 0.88), turn=(0.66, 0.72, 0.94), sky_fill=0.35,
             lit_lo=(0.8, 0.83, 0.94), lit=(0.965, 0.965, 0.965), hi=(1.02, 1.0, 0.965),
             band=(0.8, 0.82, 0.95), band_w=0.06, tint_var=0.35, strokes=0.012,
             crisp=(0.12, 0.55), lost=(0.0, 1.0), base_dark=0.12, base_h=0.12,
             detail=0.8, sep=0.9, islands=0.012, side_gate=(-0.1, 0.15), rim=1.0, rim_px=2.2, rim_col=(1.3, 1.24, 1.1), rim_reach=0.28)


def paint_s01(R, w, h, base_y, Hc, sun_px, sun_dir, seed=0, **kw):
    """Painted colour from the volumetric fields: broad lit plane with a crisp brushy terminator, a
    saturated cool shadow core lifted by sky fill on up-facing planes, a lilac turning band, a hot rim
    on the sun-facing silhouette near the sun, crisp lit / lost shaded silhouettes."""
    p = dict(PAINT)
    p.update(kw)
    x0, y0, x1, y1 = R['box']
    A = R['A']
    ia = 1.0 / np.maximum(A, 1e-4)
    S = R['sun'] * ia
    Am = R['amb'] * ia
    sc = w / 1920.0
    hh, ww = A.shape
    lam = np.clip((R['lam'] + p['lam_wrap']) / (1 + p['lam_wrap']), 0, 1)
    Sl = S * (p['lam_floor'] + (1 - p['lam_floor']) * lam)
    v = np.clip(p['k_sun'] * Sl + p['k_amb'] * Am, 0, None)
    tk = p['tone_k']
    v = (1.0 - np.exp(-v * tk)) / (1.0 - math.exp(-tk))
    v = np.clip(v, 0, None) ** p['gamma']
    if p['pre_kuwa']:
        v = KK.kuwahara(np.repeat(v[..., None], 3, -1), max(1, int(round(p['pre_kuwa'] * sc))), q=6.0)[..., 0]
    wn = KK._noise(ww, hh, max(ww / (60.0 * sc), 3), seed + 4, 4) * p['warp']
    wn += KK._noise(ww, hh, max(ww / (14.0 * sc), 6), seed + 5, 2) * p['warp'] * 0.4
    v2 = v + wn
    t, e = p['term'], p['term_e']
    lit = _ss(t - e, t + e, v2)
    if p['side_gate']:
        # the shadow flank of the big form never carries full-white lit planes (no cotton balls in the shade)
        lb = cv2.GaussianBlur(R['lam'], (0, 0), 20 * sc)
        lit = lit * _ss(p['side_gate'][0], p['side_gate'][1], lb)
    if p['islands']:
        # clean the lit plane: no lit specks floating in the shadow, no dark pits inside the light
        r_ = max(int(round(p['islands'] * Hc)), 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))
        lo = cv2.morphologyEx(lit, cv2.MORPH_OPEN, ker)
        lit = cv2.morphologyEx(lo, cv2.MORPH_CLOSE, ker)
        lit = cv2.GaussianBlur(lit, (0, 0), 0.7 * sc + 0.3)
        v2 = np.where(lit > 0.5, np.maximum(v2, t + e), np.minimum(v2, t - e))
    # shadow: deep core -> shade -> turning tone; up-facing shaded planes are lifted by the sky
    u = np.clip(v2 / t, 0, 1)
    sh = np.where((u < 0.6)[..., None], _lerp3(p['deep'], p['shade'], np.clip(u / 0.6, 0, 1)),
                  _lerp3(p['shade'], p['turn'], np.clip((u - 0.6) / 0.4, 0, 1)))
    sh = sh + (np.asarray(p['turn'], F32) - sh) * (p['sky_fill'] * np.clip(Am - 0.4, 0, 1))[..., None]
    # lit: lit_lo -> lit -> hi
    lw = np.clip((v2 - t) / (1.0 - t), 0, 1)
    if p['detail']:
        # secondary modelling inside the lit plane: the local (per-head) sun term vs its broad average
        Sb = cv2.GaussianBlur(S, (0, 0), 30 * sc)
        sdet = np.clip(S / np.maximum(Sb, 1e-3) - 1.0, -1, 1)
        sdet = KK.kuwahara(np.repeat(sdet[..., None], 3, -1), max(1, int(round(4 * sc))), q=6.0)[..., 0]
        lw = np.clip(lw + p['detail'] * sdet, 0, 1)
    sep = np.zeros_like(v)
    if p['sep']:
        # head separation: a surface lying just above a NEARER head's top outline is in its crevice
        Dp = R['depth']
        Dp = np.where(A > 0.2, Dp, Dp.max())
        for k in (4, 8, 13, 19):
            kk = max(int(round(k * sc)), 1)
            below = np.empty_like(Dp)
            below[:-kk] = Dp[kk:]
            below[-kk:] = Dp[-kk:]
            sep = np.maximum(sep, _ss(0.01 * Hc, 0.05 * Hc, Dp - below) * (1 - k / 25.0))
        sep = cv2.GaussianBlur(sep, (0, 0), 1.2 * sc + 0.3) * _ss(0.2, 0.6, A)
        lw = np.clip(lw - p['sep'] * sep, 0, 1)
    lc = np.where((lw < 0.5)[..., None], _lerp3(p['lit_lo'], p['lit'], lw / 0.5),
                  _lerp3(p['lit'], p['hi'], (lw - 0.5) / 0.5))
    # a thin lilac band just on the shadow side of the terminator (separates the lit crescent)
    band = _ss(t - e - p['band_w'], t - e, v2) * (1 - lit)
    sh = sh + (np.asarray(p['band'], F32) - sh) * (band * 0.6)[..., None]
    col = sh * (1 - lit[..., None]) + lc * lit[..., None]
    if p['post_kuwa']:
        col = KK.kuwahara(col, max(1, int(round(p['post_kuwa'] * sc))), q=6.0)
    if p['tint_var']:
        tn = KK._noise(ww, hh, max(ww / (160.0 * sc), 2), seed + 31, 3)
        k_ = np.clip(tn * 0.5 + 0.5, 0, 1)[..., None]
        warm = np.array([1.0, 0.99, 0.965], F32)
        cool = np.array([0.965, 0.98, 1.02], F32)
        col = col * (1 + p['tint_var'] * ((warm - 1) * k_ + (cool - 1) * (1 - k_)) * 2)
    if p['strokes']:
        st = KK._noise(ww, hh, max(ww / (9.0 * sc), 6), seed + 41, 2, stretch=5.0, angle=-35)
        col = col * (1 + p['strokes'] * st[..., None])
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    if p['base_dark']:
        bd = 1 - _ss(0, p['base_h'] * Hc, base_y - ys)
        col = col * (1 - p['base_dark'] * bd[..., None])
    # silhouette alpha: crisp where lit, lost where shaded
    litm = _ss(0.25, 0.7, S)
    a0 = p['crisp'][0] * litm + p['lost'][0] * (1 - litm)
    a1 = p['crisp'][1] * litm + p['lost'][1] * (1 - litm)
    Ae = np.clip((A - a0) / np.maximum(a1 - a0, 1e-3), 0, 1)
    Ae = Ae * Ae * (3 - 2 * Ae)
    # rim: hot lining on the sun-facing silhouette, strongest near the sun
    if p['rim']:
        Ab = cv2.GaussianBlur(Ae, (0, 0), 4.0 * sc + 1)
        gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
        gl = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = -gx / gl, -gy / gl
        sd = np.asarray(sun_dir, F32)
        sd = sd / np.linalg.norm(sd)
        face = np.clip(nx * sd[0] + ny * sd[1], 0, 1)
        d = np.hypot(xs - sun_px[0], ys - sun_px[1]) / Hc
        near = np.exp(-(d / p['rim_reach']) ** 2)
        rb = max(p['rim_px'] * sc, 0.8)
        er = cv2.erode(Ae, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(rb) + 1, 2 * int(rb) + 1)))
        edge = np.clip(Ae - er, 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 0.6 * sc + 0.3)
        k = edge * np.clip(near * (0.35 + 0.65 * np.clip(-ny, 0, 1)) + 0.35 * face ** 2 * np.exp(-d / 0.6), 0, 1)
        col = col + (np.asarray(p['rim_col'], F32) - col) * np.clip(k * p['rim'], 0, 1)[..., None]
    out = np.zeros((h, w, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = Ae
    out[..., :3] = KK._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


def build_lobes(rng, cx, base_y, Hc, eye, dist, w, L, detail=1.0, main=None, custom=None):
    """Lobe array (camera space) for the hero tower."""
    X = cx - w / 2.0
    Yb = base_y - eye
    rows = []
    if custom is not None:
        return custom_lobes(rng, X, Yb, dist, Hc, L, **custom)
    wd = 0.5 * Hc
    sizes = tuple(s * detail for s in (0.13, 0.065, 0.032))
    mk = dict(lean=-0.035, prof=_prof_main(rng.uniform(0, 6.28)), n_heads=18, r_rng=(0.55, 0.95), recede=0.45,
              shrink=0.0, sizes=sizes, dens=0.95, wobble=0.1)
    mk.update(main or {})
    rows.append(K3.head_lobes(rng, X, Yb, dist, wd, Hc, sun3=L, **mk))
    # lower turret on the shadow (right) side, a little nearer
    rows.append(K3.head_lobes(rng, X + 0.2 * Hc, Yb, dist - 0.05 * Hc, 0.3 * Hc, 0.34 * Hc, lean=0.06, sun3=L,
                              n_heads=10, r_rng=(0.45, 0.8), recede=0.3, shrink=0.1, sizes=sizes, dens=0.9))
    # low left shoulder, nearer still
    rows.append(K3.head_lobes(rng, X - 0.2 * Hc, Yb, dist - 0.1 * Hc, 0.26 * Hc, 0.26 * Hc, lean=-0.05, sun3=L,
                              n_heads=8, r_rng=(0.45, 0.8), recede=0.3, shrink=0.1, sizes=sizes, dens=0.9,
                              prof=K3.heap_prof(rng.uniform(0, 6.28))))
    return np.concatenate(rows, 0)


HERO = dict(custom=dict(hw0=0.19, hw1=0.08, taper=1.0, side_r=(0.25, 0.6), crown=0.1, lean=0.03,
                        anvil=dict(left=0.08, right=0.22, thick=0.07, neck=3.0, col_hw=0.07, top=1.0),
                        turrets=((0.16, 0.3, 0.075, -0.1), (-0.15, 0.2, 0.065, -0.12))),
            seed=5, render=dict(form=0.8, form_blur=8.0), sun_dir=(-0.9, -0.45), sun_z=0.3,
            paint=dict(term=0.42, detail=1.4, sep=1.2, islands=0.02, side_gate=(-0.35, -0.05), crisp=(0.35, 0.5), rim_px=4.0, rim_col=(1.5, 1.4, 1.2), rim_reach=0.35,
                       deep=(0.3, 0.44, 0.76), shade=(0.46, 0.59, 0.87), turn=(0.7, 0.78, 0.93),
                       lit=(0.97, 0.965, 0.94), hi=(1.02, 0.995, 0.94)),
            florets=dict(density=2.0, r=(1.5, 7.0)), sun_at=(0.03, 0.0))


def hero(pw, ph, cx, base_y, Hc, W, hz_y, **kw):
    k = dict(HERO)
    import os
    if os.environ.get('S01_HERO'):          # dev-only override (unset in production renders)
        kw = dict(eval('dict(' + os.environ['S01_HERO'] + ')'), **kw)
    for key, v in kw.items():
        if isinstance(v, dict) and isinstance(k.get(key), dict):
            k[key] = dict(k[key], **v)
        else:
            k[key] = v
    return tower(pw, ph, cx, base_y, Hc, W, hz_y, **k)


def tower(pw, ph, cx, base_y, Hc, W, hz_y, seed=41, sun_z=0.15, max_dim=300, sun_dir=(-0.62, -0.78),
          persp=4.0, render=None, paint=None, dbg=None, main=None, florets=None, custom=None, sun_at=None):
    """Returns (rgba plate (ph, pw, 4), sun position in plate px)."""
    top = base_y - Hc
    sun = np.array([cx - 0.1 * Hc, top + 0.04 * Hc], np.float32)
    eye = base_y + 0.05 * ph
    dist = persp * Hc
    L = KK._sun_vec(None, sun_dir, sun_z, None)
    rng = np.random.default_rng(seed)
    E = build_lobes(rng, cx, base_y, Hc, eye, dist, pw, L, main=main, custom=custom)
    rk = dict(fine_amp=0.5, erode=(0.8, 8.0, 3, 0.8), seed=seed)
    rk.update(render or {})
    R = K3.vol_render(E, pw, ph, dist, pw / 2.0, eye, L, max_dim=max_dim, **rk)
    if dbg is not None:
        dbg['R'] = R
    # the sun sits right on the crown's upper-left shoulder (partly hidden behind it)
    x0, y0, x1, y1 = R['box']
    Ab = R['A'] > 0.5
    rows_ = np.nonzero(Ab.any(1))[0]
    ytop = rows_[0] + y0
    sa = sun_at or (0.06, 0.02)
    ys_ = int(ytop + sa[0] * Hc)
    cols_ = np.nonzero(Ab[ys_ - y0])[0]
    sun = np.array([cols_[0] + x0 + sa[1] * Hc, ys_], np.float32)
    if paint is not None and paint.get('legacy'):
        pk = dict(k_sun=0.55, ramp='s01', seed=seed, base_y=base_y, base_dark=0.1, base_h=0.1)
        pk.update(paint)
        pk.pop('legacy')
        P = K3.paint_vol(R, pw, ph, **pk)
    else:
        P = paint_s01(R, pw, ph, base_y, Hc, sun, sun_dir, seed=seed, **(paint or {}))
    fk = dict(sun_dir=sun_dir, seed=seed)
    fk.update(florets or {})
    K3.edge_florets(P, **fk)
    return P, sun
