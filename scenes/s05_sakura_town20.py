"""s05_sakura round 20: far-bank town post-pass (reviewer: 'left-bank buildings still flat CG boxes with uniform
roofs -> painted roof-tile value variation, window reflections with sky gradients, AC units, laundry and small
signage clutter (yn_07 detail density), warm-lit sun-facing facades, far buildings fading into blue haze').

Works on the supersampled environment buffer (straight RGB + material / face / depth maps) before it is split
into plates, so everything moves with the town's exact parallax.
  * roofs (pitched planes + flat tops): per-roof hue family (navy / slate / brown / dark green / rust) at the
    painted value, patchy weathered-tile mottle in world units, a few moss / rust streaks, sky sheen gradient;
  * facades: sunlit faces warmed toward cream-apricot, shade faces cooled toward blue-violet;
  * clutter on the river-facing facades of the nearer blocks: AC units (pale box + dark fan disc + pipe),
    laundry on the balconies (short runs of coloured cloth), small plain signboards and vertical banner panels
    (colour + light border, no fake lettering), window glints with a vertical sky gradient;
  * distance: blocks beyond ~200 m sink into a clean blue aerial haze.
"""
import math

import numpy as np
import cv2

ROOF_HUES = np.array([
    (0.3, 0.36, 0.56),     # navy-slate
    (0.42, 0.44, 0.52),    # grey
    (0.5, 0.36, 0.3),      # brown
    (0.3, 0.42, 0.4),      # dark green-teal
    (0.58, 0.32, 0.26),    # rust red
    (0.36, 0.38, 0.62),    # blue
], np.float32)


def _vn(rng, h, w, cell):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]


def town_pass(out, mat, bid, face, uo, Yo, so, layer, Zo, f, ss, W, B_typ, max_obj=110, var_lim=0.02, seed=2020):
    rng = np.random.default_rng(seed)
    Hs, Ws = mat.shape
    rgb = out[..., :3]
    town = (mat == 5) & (layer == 0)
    if not town.any():
        return
    af = np.abs(face)
    roof = town & ((af == 4) | (face == 2))
    fac = town & ((af == 1) | (af == 3))
    lum = rgb.mean(-1)
    # ------------------------------------------------------------------ roofs
    nb = int(bid.max()) + 1
    hue_i = rng.integers(0, len(ROOF_HUES), nb)
    tone = rng.uniform(-0.07, 0.07, nb).astype(np.float32)
    bsafe = np.clip(bid, 0, nb - 1)
    if roof.any():
        ys, xs = np.nonzero(roof)
        b_ = bsafe[ys, xs]
        c = rgb[ys, xs]
        l_ = c.mean(-1, keepdims=True)
        h_ = ROOF_HUES[hue_i[b_]]
        hl = h_.mean(-1, keepdims=True)
        # keep the painted value, swap the hue family (blend 60 %)
        tgt = h_ / np.maximum(hl, 1e-3) * l_
        c2 = c + (tgt - c) * 0.6
        c2 = c2 * (1.0 + tone[b_])[:, None]
        # weathered tile mottle in world units (patches ~0.8 m and ~0.25 m)
        uu, ssv = uo[ys, xs], so[ys, xs]
        m1 = np.sin(uu * 3.1 + ssv * 1.7 + b_ * 1.3) * np.sin(ssv * 2.3 - uu * 1.1 + b_ * 0.7)
        m2 = np.sin(uu * 11.0 + b_) * np.sin(ssv * 9.0 + 2.0 * b_)
        mot = 0.06 * m1 + 0.035 * m2
        c2 = c2 * (1.0 + mot)[:, None]
        # moss / rust streaks on a few roofs
        st = np.sin(ssv * 4.0 + uu * 0.5 + b_ * 2.1)
        streak = np.clip((st - 0.75) / 0.2, 0, 1) * (hue_i[b_] % 3 == 0)
        c2 = c2 + (np.array([0.42, 0.42, 0.3], np.float32) * l_ * 1.4 - c2) * (0.3 * streak)[:, None]
        rgb[ys, xs] = c2
        # reveal the painted tile courses: local contrast on the roof planes only
        rb = cv2.GaussianBlur(rgb, (0, 0), 2.5 * ss / 2)
        rk = roof.astype(np.float32)[..., None]
        rgb[:] = rgb + (rgb - rb) * 1.6 * rk
    # ------------------------------------------------------------------ facades: warm lit / cool shade
    if fac.any():
        ftone = rng.uniform(-0.06, 0.06, nb).astype(np.float32)
        rgb[:] = rgb * (1.0 + ftone[bsafe] * fac)[..., None]
        lit = np.clip((lum - 0.6) / 0.25, 0, 1) * fac
        shd = np.clip((0.5 - lum) / 0.25, 0, 1) * fac
        rgb[:] = rgb * (1 + lit[..., None] * np.array([0.07, 0.02, -0.07], np.float32))
        rgb[:] = rgb + (np.array([0.3, 0.32, 0.5], np.float32) * lum[..., None] * 1.5 - rgb) * (0.12 * shd)[..., None]
    # ------------------------------------------------------------------ clutter on the nearer facades
    near = fac & (Zo < 230.0) & (Zo > 1.0)
    ys, xs = np.nonzero(near)
    if len(ys):
        n_try = int(min(len(ys), 2600))
        pick = rng.choice(len(ys), n_try, replace=False)
        occ = np.zeros((Hs, Ws), bool)
        LAUNDRY = np.array([(0.96, 0.96, 0.94), (0.55, 0.7, 0.92), (0.95, 0.72, 0.78), (0.95, 0.9, 0.62),
                            (0.35, 0.45, 0.7), (0.9, 0.55, 0.4)], np.float32)
        SIGN = np.array([(0.85, 0.2, 0.18), (0.16, 0.45, 0.35), (0.95, 0.72, 0.2), (0.2, 0.35, 0.7),
                         (0.95, 0.95, 0.92)], np.float32)
        per_b = {}
        placed = 0
        # plain wall test: low local variance (no windows / signs / lettering under the object)
        g_ = rgb.mean(-1)
        m_ = cv2.blur(g_, (5, 5))
        var_ = cv2.blur(g_ * g_, (5, 5)) - m_ * m_
        for pi in pick:
            if placed >= max_obj:
                break
            y, x = int(ys[pi]), int(xs[pi])
            Z = float(Zo[y, x])
            mpx = f * ss / Z                                   # px per metre
            if mpx < 3.0:
                continue
            b = bid[y, x]
            fc = face[y, x]
            if per_b.get(int(b), 0) >= 4 or int(B_typ[b]) not in (0, 5):
                continue
            kind = rng.random()
            if kind < 0.4:
                w_, h_ = 0.85 * mpx, 0.6 * mpx               # AC unit
            elif kind < 0.72:
                w_, h_ = rng.uniform(1.2, 2.6) * mpx, rng.uniform(0.5, 0.8) * mpx     # laundry run
            elif kind < 0.88:
                w_, h_ = rng.uniform(0.9, 1.6) * mpx, rng.uniform(0.4, 0.7) * mpx     # small signboard
            else:
                w_, h_ = rng.uniform(0.35, 0.5) * mpx, rng.uniform(1.2, 2.0) * mpx    # vertical banner panel
            wi, hi = int(round(w_)), int(round(h_))
            if wi < 3 or hi < 2:
                continue
            x0, y0 = x - wi // 2, y - hi // 2
            if x0 < 1 or y0 < 1 or x0 + wi >= Ws - 1 or y0 + hi >= Hs - 1:
                continue
            win = (slice(y0 - 1, y0 + hi + 1), slice(x0 - 1, x0 + wi + 1))
            if not ((bid[win] == b) & (face[win] == fc) & near[win]).all() or occ[win].any():
                continue
            if var_[win].mean() > var_lim:
                continue
            occ[win] = True
            per_b[int(b)] = per_b.get(int(b), 0) + 1
            placed += 1
            base = rgb[y0:y0 + hi, x0:x0 + wi]
            lf = float(np.clip(base.mean() / 0.7, 0.55, 1.15))       # take the facade's light level
            if kind < 0.4:
                c = np.array([0.9, 0.9, 0.88], np.float32) * lf
                base[:] = c
                base[-max(1, hi // 5):] = c * 0.62                     # shadowed underside
                base[:max(1, hi // 8)] = np.minimum(c * 1.12, 1.1)     # lit top edge
                r = max(1, int(min(wi, hi) * 0.3))
                cx_, cy_ = int(wi * 0.62), hi // 2
                yy, xx = np.mgrid[0:hi, 0:wi]
                disc = ((xx - cx_) ** 2 + (yy - cy_) ** 2) <= r * r
                base[disc] = c * 0.42
                # refrigerant pipe down the wall
                px0 = x0 + max(1, wi // 8)
                y1 = min(Hs, y0 + hi + int(0.9 * mpx))
                col_ok = (bid[y0 + hi:y1, px0] == b)
                rgb[y0 + hi:y1, px0][col_ok] = np.array([0.82, 0.8, 0.76], np.float32) * lf
            elif kind < 0.72:
                # a rail line with pieces of cloth hanging from it
                xcur = 0
                while xcur < wi:
                    cw = int(max(2, rng.uniform(0.18, 0.45) * mpx))
                    ch = int(max(2, rng.uniform(0.45, 1.0) * hi))
                    cc = LAUNDRY[int(rng.integers(len(LAUNDRY)))] * lf
                    seg = base[:ch, xcur:xcur + cw]
                    seg[:] = cc
                    seg[:, -1:] = cc * 0.8                             # fold shade
                    xcur += cw + int(rng.integers(0, max(1, cw // 2) + 1))
                base[0:1] = np.array([0.7, 0.72, 0.75], np.float32) * lf
            else:
                c = SIGN[int(rng.integers(len(SIGN)))] * min(lf * 1.05, 1.1)
                base[:] = c
                bw = max(1, int(min(wi, hi) * 0.12))
                lc = np.minimum(c * 0.4 + 0.6, 1.05)
                base[:bw] = lc
                base[-bw:] = lc
                base[:, :bw] = lc
                base[:, -bw:] = lc
                # a plain lighter band (a painted panel, not fake lettering)
                if kind < 0.88 and hi > 5:
                    base[hi // 2 - max(1, hi // 10):hi // 2 + max(1, hi // 10), bw + 1:wi - bw - 1] = \
                        c + (lc - c) * 0.35
            # cast shadow of the object on the wall (to the lower right, away from the upper-left sun)
            sh = max(1, int(0.08 * mpx))
            sy0, sx0 = y0 + hi, x0 + sh
            if sy0 + sh < Hs and sx0 + wi < Ws:
                wall = rgb[sy0:sy0 + sh, sx0:sx0 + wi]
                okm = (bid[sy0:sy0 + sh, sx0:sx0 + wi] == b)[..., None]
                wall[:] = np.where(okm, wall * np.array([0.78, 0.78, 0.86], np.float32), wall)
    # ------------------------------------------------------------------ distance: blue aerial haze
    far = town & (Zo > 180.0)
    if far.any():
        k = np.clip((Zo - 180.0) / 220.0, 0, 1) * 0.45 * far
        hz = np.array([0.74, 0.82, 0.95], np.float32)
        rgb[:] = rgb + (hz - rgb) * k[..., None]
    out[..., :3] = np.clip(rgb, 0, 1.25)
