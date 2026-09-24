"""s07_comet_night - Night sky over a mountain lake (Your Name homage).

Round 7: Your-Name dust tail with a spectral spread (cyan core -> teal -> gold -> magenta -> violet) plus a
separate straight blue ion tail (s07_comet_night_v5), warm kataware-doki horizon glow, crisp broken Milky Way
dust lanes, painted rock/snow structure (outcrops, strata, facets, spur linework) and a warm twilight rim on
the left-facing ridges against the cyan comet rim.

One luminous comet high in the sky with the Your-Name twin tail: a long, gently curving blue-white main
tail and - where a fragment has peeled off the nucleus - a cyan secondary tail with its own ion core
splitting away at ~10 degrees, both wrapped in a soft cyan halo, with glinting fragment sparkle. A Milky Way painted as granular star clouds with soft, feathered, warm-edged dark rifts. Snow and
rock ranges painted as big lit/shadow planes (curved spurs, altitude snowline with V-shaped couloir
tongues, bare rock flanks, comet-facing crest rims, aerial falloff: the distant range paler and bluer),
forested foothills as overlapping serrated rows with mist pooling between them, a low clouds2
stratocumulus bank drifting in the valley, a
lakeside town (torii, school with clock tower, fire tower, utility poles and wires) with warm lights, low
mist, and the whole sky mirrored in the rippling lake.
Camera: slow tilt down from the sky to the reflection with a slight push-in and a lateral truck that
separates the planes (sky still; distant range, far range, forested ridges, town/shore and reeds each
slide progressively faster); the tail slowly lengthens and the reflected lights shimmer.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C, fx as F, clouds2 as K2  # noqa: E402
import s07_comet_night_paint as P  # noqa: E402
import s07_comet_night_rock as R  # noqa: E402
import s07_comet_night_v3 as V  # noqa: E402
import s07_comet_night_v4 as V4  # noqa: E402
import s07_comet_night_mtn as MT  # noqa: E402
import s07_comet_night_terrain as TR  # noqa: E402
import s07_comet_night_hills as HL  # noqa: E402
import s07_comet_night_v5 as V5  # noqa: E402

DURATION = 5.5


def _hex(h):
    return C.hex2rgb(h)


def _rot(d, deg):
    a = math.radians(deg)
    return (d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a))


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        s = W / 1920.0
        self.s = s
        Wp = int(round(W * 1.12))
        self.tilt = 0.52 * H
        my = int(0.02 * H)
        Hp = int(my + 1.0 * H + self.tilt + 0.04 * H)
        self.Wp, self.Hp, self.my = Wp, Hp, my
        y_h = my + 1.06 * H                        # waterline (far shore) in plate px
        self.y_h = y_h
        self.truck = 0.07 * W                       # lateral truck amplitude (town plane)
        xs, ys = C.grid(Wp, Hp)

        # comet geometry (needed early for lighting)
        self.head = (Wp / 2 + 0.13 * W, my + 0.63 * H)
        hx, hy = self.head

        # ------------------------------------------------------------------ sky
        sky = P.sky_backdrop(Wp, Hp, y_h, [
            (0.0, '#030817'), (0.22, '#06112a'), (0.45, '#0a1b42'), (0.66, '#102858'),
            (0.8, '#193468'), (0.9, '#273d74'), (0.96, '#38427c'), (1.0, '#474882')], seed=1)
        yy = ys[..., None] / y_h
        veil = np.exp(-((yy - 0.6) / 0.2) ** 2) * np.array([0.05, 0.0, 0.08], np.float32)
        sky = sky + veil * (0.6 + 0.4 * P.fbm_lowres(Wp, Hp, 2, 3, 5, q=8)[..., None])
        # warm kataware-doki glow hugging the horizon (strongest left of centre, away from the comet)
        tw_add, tw_mix = V5.twilight(Wp, Hp, y_h, H, xs, ys, 0.3 * Wp, seed=8)
        sky = sky * (1 - tw_mix[..., None] * np.array([0.3, 0.55, 0.8], np.float32)) + tw_add
        tg = np.exp(-((xs - 0.27 * Wp) / (0.17 * Wp)) ** 2) * np.exp(-np.clip(y_h - ys, 0, None) / (0.05 * H)) *             (ys < y_h + 2)
        sky += tg[..., None] * np.array([0.22, 0.11, 0.04], np.float32)
        # cool cyan skyglow around / below the comet
        cg = np.exp(-((xs - hx) / (0.28 * Wp)) ** 2) * np.exp(-((ys - (hy + 0.05 * H)) / (0.3 * H)) ** 2)
        sky += cg[..., None] * np.array([0.0, 0.05, 0.09], np.float32)

        # Milky Way rising from the right-hand horizon, clear of the comet
        mw, dust, dens = V4.milky_way4(Wp, Hp, (Wp / 2 + 0.43 * W, y_h + 0.03 * H), (Wp / 2 + 0.02 * W, my - 0.15 * H),
                                       width=0.12 * H, seed=11, strength=1.0, bend=0.03, s=s)
        ext = 1 - 0.8 * C.smoothstep(y_h - 0.5 * H, y_h, np.arange(Hp, dtype=np.float32))   # extinction
        mw *= ext[:, None, None]
        dens *= ext[:, None] ** 2
        sky = sky * (1 - 0.12 * dust[..., None]) + mw

        st, phase = P.stars(Wp, Hp, 21, dens, n_base=1300)
        st += V.fine_stars(Wp, Hp, 23, dens, int(170000 * Wp * Hp / (1920 * 1080)))
        st += R.big_glints(Wp, Hp, 29, 12)
        fade = 1 - C.smoothstep(y_h - 0.25 * H, y_h, np.arange(Hp, dtype=np.float32)) * 0.75
        st *= fade[:, None, None]
        self.stars = st.astype(np.float32)
        self.phase = phase
        del mw, st

        # ------------------------------------------------------------------ comet
        L = 1.2 * H
        d1 = (-0.78, -0.63)
        n1 = math.hypot(*d1)
        d1 = (d1[0] / n1, d1[1] / n1)
        tb, tw0, tw1 = 0.16, 0.0035, 0.11
        t1, u1, vn1, b1 = V5.comet_tail5(Wp, Hp, self.head, d1, L, W, s, bend=tb, w0=tw0, w1=tw1, seed=3)
        t1 += V5.tail_particles(Wp, Hp, self.head, d1, L, W, s, tb, tw0, tw1, int(1800 * s + 300), 31)
        # Your-Name split: a fragment just behind the nucleus; the thin straight blue ion tail diverges
        # from the nucleus on the leading side
        self.frag = (hx + d1[0] * 0.035 * L, hy + d1[1] * 0.035 * L)
        d2 = _rot(d1, -14.0)
        t2 = V5.ion_tail(Wp, Hp, self.head, d2, 0.75 * L, W, s)
        tail = t1 + t2
        # faint cool halo enveloping the tail (wide, low, never clipping)
        tl_l = tail.max(-1)
        halo = F.fast_blur(tl_l, 0.02 * W) * 0.3 + F.fast_blur(tl_l, 0.06 * W) * 0.25
        tail = tail + halo[..., None] * np.array([0.12, 0.3, 0.55], np.float32)
        del tl_l, halo, t2
        head_img = V.comet_head(Wp, Hp, self.head, s, 1.0)
        head_img += V.comet_head(Wp, Hp, self.frag, s, 0.28, col=(0.6, 1.0, 0.95))
        rng = np.random.default_rng(404)
        sp = []
        for k in range(4):
            q = rng.uniform(0.0, 0.16) ** 1.3 * 1.8
            base = (self.frag[0] + d2[0] * q * L, self.frag[1] + d2[1] * q * L)
            off = rng.normal(0, 0.004 * W)
            sp.append((base[0] - d2[1] * off, base[1] + d2[0] * off, rng.uniform(0.4, 1.0),
                       (0.75, 1.0, 0.95) if rng.random() < 0.6 else (0.85, 0.9, 1.0)))
        # the Your-Name split: a few tiny fragments just ahead of / beside the nucleus
        fwd = (-d1[0], -d1[1])
        side = (-d1[1], d1[0])
        for (a_, b_, it) in ((0.022, 0.006, 1.0), (0.034, -0.004, 0.75), (0.014, 0.016, 0.6)):
            sp.append((hx + (fwd[0] * a_ + side[0] * b_) * W, hy + (fwd[1] * a_ + side[1] * b_) * W, it,
                       (0.85, 1.0, 0.97)))
        self.n_front = 3
        self.sparks = sp
        dcen = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2) / W
        cglow = np.exp(-dcen / 0.12)[..., None] * np.array([0.03, 0.06, 0.11], np.float32)
        fl_ = F.anime_flare(Wp, Hp, hx, hy, intensity=0.35, tint=(0.6, 0.8, 1.0), rays=4,
                            ray_len=0.03, ghosts=0.0, halo=0.4, streak=0.2, glow=0.5, rot=0.4)
        self.comet_static = (head_img + cglow + fl_).astype(np.float32)
        del head_img, cglow, fl_, dcen
        # tail crop (per-frame lengthening + flowing streamers)
        bm = (tail.max(-1) > 0.003)
        rws, cls = np.nonzero(bm.any(1))[0], np.nonzero(bm.any(0))[0]
        tx0, ty0, tx1, ty1 = cls.min(), rws.min(), cls.max() + 1, rws.max() + 1
        self.tbox = (tx0, ty0)
        self.tail = np.ascontiguousarray(tail[ty0:ty1, tx0:tx1]).astype(np.float32)
        self.tail_aux = np.dstack([u1[ty0:ty1, tx0:tx1], vn1[ty0:ty1, tx0:tx1],
                                   b1[ty0:ty1, tx0:tx1]]).astype(np.float32)
        full_tail = tail
        del u1, vn1, b1
        self.sky = sky.astype(np.float32)

        # ------------------------------------------------------------------ land layers
        ss = 2
        xs2 = np.arange(Wp * ss, dtype=np.float32) / ss
        xf = xs2 / Wp
        Hl = int(y_h + 0.01 * H) + 2
        self.Hl = Hl
        b0 = int(y_h - 0.42 * H)
        self.b0 = b0
        cometL = (hx, hy)
        lights = []

        def env(pts):
            p = np.array(pts, np.float32)
            return np.interp(xf, p[:, 0], p[:, 1]).astype(np.float32)

        def ridge_rgba(top, cols, fade_px):
            ysl = np.arange(Hl, dtype=np.float32)[:, None]
            msk = np.clip(ysl - top[None, :] + 0.5, 0, 1)
            msk = cv2.resize(msk, (Wp, Hl), interpolation=cv2.INTER_AREA)
            top1 = top.reshape(Wp, ss).min(1)
            return P.layer_shade(Wp, Hl, top1, msk, cols[0], cols[1], fade_px)

        def topline(h):
            return y_h - h

        layers = []          # (rgba band, parallax factor)

        # snow ranges: an eroded terrain ray-cast from the lake and painted (see s07_comet_night_terrain):
        # faint back range, far centre peaks, near massif + right ridge -> three parallax plates
        cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'cache')
        for (pl, fpar) in TR.range_plates(Wp, H, W, y_h, b0, Hl, cometL, cache_dir=cache):
            full_ = np.zeros((Hl, Wp, 4), np.float32)
            full_[b0:Hl] = pl
            layers.append((full_, fpar))

        # soft mist band lying between the ranges and the forested foothills (full width, low contrast,
        # faintly lit from above), plus a couple of torn wisps drifting across the lower slopes
        layers.append((TR.mist_layer(Wp, Hl, y_h, H, s, hx, seed=61), 0.42))
        self.cloud_layer = len(layers) - 1
        self.cloud_drift = 0.006 * W

        # mid ridges: fully forested, trees stamped one by one (their tips form the serrated skyline)
        mid_h = env([(0, 0.15), (0.1, 0.17), (0.2, 0.12), (0.3, 0.075), (0.4, 0.055), (0.5, 0.045),
                     (0.6, 0.04), (0.68, 0.05), (0.78, 0.09), (0.88, 0.13), (1.0, 0.15)])
        mid_h += 0.018 * P.fbm1d(xf, 9, 5, 201, ridged=True)
        mid_top = y_h - mid_h * H
        mid_top1 = mid_top.reshape(Wp, ss).min(1)
        # three overlapping rows of forested foothills, mist pooling between them
        h2 = np.maximum(mid_h * 0.62 + 0.004 + 0.012 * P.fbm1d(xf, 6, 3, 211), 0.012)
        h3 = np.maximum(mid_h * 0.3 + 0.002 + 0.008 * P.fbm1d(xf, 8, 3, 213), 0.006)
        rows_ = [mid_top, np.maximum(y_h - h2 * H, mid_top + 0.01 * H), np.maximum(y_h - h3 * H, mid_top + 0.02 * H)]
        mid = HL.forest_hills2(Wp, Hl, rows_, ss, s, H, cometL, 205, y_h, 0.0105 * H,
                               dict(snow_lit=_hex('#48679f'), snow_sh=_hex('#1c306a'), snow_gully=_hex('#1a2d66'),
                                    forest_lit=_hex('#172d63'), forest_sh=_hex('#0c1a45'), forest_base=_hex('#081236')),
                               _hex('#35599a'), mist_amt=0.32, back_haze=0.42, rim_col=(0.42, 0.66, 0.95))
        # pylon on the right ridge (blinking red light) + wires sagging across the valley
        tx = 0.86 * Wp
        ti = int(tx)
        ttop = mid_top1[ti] - 0.035 * H
        self.tower_light = (tx, ttop)
        mb = mid_top1[ti] + 0.004 * H
        pw_ = 0.005 * W
        tw = [[(tx - pw_, mb), (tx - pw_ * 0.25, ttop + 0.01 * H), (tx, ttop), (tx + pw_ * 0.25, ttop + 0.01 * H),
               (tx + pw_, mb)],
              [(tx - pw_ * 0.7, mb - 0.01 * H), (tx + pw_ * 0.7, mb - 0.01 * H)],
              [(tx - pw_ * 1.6, ttop + 0.012 * H), (tx + pw_ * 1.6, ttop + 0.012 * H)],
              [(tx - pw_ * 1.2, ttop + 0.02 * H), (tx + pw_ * 1.2, ttop + 0.02 * H)],
              [(tx - pw_ * 0.7, mb - 0.01 * H), (tx + pw_ * 0.45, ttop + 0.02 * H)],
              [(tx + pw_ * 0.7, mb - 0.01 * H), (tx - pw_ * 0.45, ttop + 0.02 * H)]]
        for dy_ in (0.012, 0.02):
            for sx in (-1, 1):
                p0 = (tx + sx * pw_ * 1.5, ttop + dy_ * H)
                p1 = (tx - 0.14 * Wp, y_h - 0.03 * H + dy_ * H * 0.3) if sx < 0 else (Wp + 5, ttop + 0.02 * H)
                tw.append([tuple(p) for p in C.catenary(p0, p1, 0.012 * H, 48)])
        tmask = P.lines_mask(Wp, Hl, tw, 0.9 * s + 0.3)
        mid[..., :3] = C.over(mid[..., :3], _hex('#0c1844'), tmask)
        mid[..., 3] = np.maximum(mid[..., 3], tmask)
        layers.append((mid, 0.55))
        # luminous mist lying over the shoreline: behind the town, in front of the far forest
        layers.append((HL.shore_mist(Wp, Hl, y_h, H, s, hx, 0.27 * Wp, seed=71), 0.8))

        # near shore ridge (forest either side of the town) + the town
        near_h = env([(0, 0.07), (0.05, 0.06), (0.09, 0.02), (0.12, 0.008), (0.43, 0.007), (0.47, 0.012),
                      (0.55, 0.015), (0.62, 0.012), (0.7, 0.025), (0.8, 0.045), (0.9, 0.07), (1.0, 0.085)])
        near_h += 0.006 * P.fbm1d(xf, 20, 3, 301)
        near_top = y_h - near_h * H
        forest_ok = lambda x: 0.0 if 0.1 * Wp < x < 0.44 * Wp else 1.0  # noqa: E731
        near_rows = [near_top, np.maximum(y_h - near_h * 0.45 * H, near_top + 0.012 * H)]
        near = HL.forest_hills2(Wp, Hl, near_rows, ss, s, H, cometL, 305, y_h, 0.0125 * H,
                                dict(snow_lit=_hex('#3f5a96'), snow_sh=_hex('#1a2e66'), snow_gully=_hex('#152858'),
                                     forest_lit=_hex('#172c62'), forest_sh=_hex('#0c1a48'), forest_base=_hex('#081338')),
                                _hex('#2a4a88'), mist_amt=0.25, back_haze=0.3, rim_col=(0.3, 0.5, 0.8), dens=forest_ok,
                                forest_bias=0.25)
        land = near[..., :3].copy()
        land_a = near[..., 3].copy()
        near_top = near_top  # supersampled, used by the town / pole code below

        # ------------------------------------------------------------------ town
        rng = np.random.default_rng(77)
        win_rects = []
        base = y_h - 0.004 * H
        sc_ = 1.6
        rows = [(base, 1.0), (base - 0.012 * H, 0.8), (base - 0.022 * H, 0.64)]
        polys_back = []
        house_rows = []          # per row (back -> front): walls, roofs, lit roof slopes, roof edge lines
        win_pal = [(1.0, 0.72, 0.36), (1.0, 0.82, 0.52), (1.0, 0.6, 0.26), (1.0, 0.9, 0.72), (0.72, 0.88, 1.0),
                   (1.0, 0.52, 0.3)]
        win_p = np.array([0.28, 0.24, 0.16, 0.14, 0.1, 0.08])
        for ri, (by, scl) in enumerate(rows[::-1]):
            scl *= sc_
            walls_, roofs_, lit_, edges_ = [], [], [], []
            x = (0.11 + 0.02 * ri) * Wp
            while x < (0.43 - 0.03 * (2 - ri)) * Wp:
                if 0.275 * Wp < x < 0.345 * Wp and ri == 1:      # leave room for the school
                    x += 0.01 * Wp
                    continue
                w = rng.uniform(0.009, 0.017) * W * scl
                h = rng.uniform(0.0055, 0.009) * H * scl
                two = rng.random() < 0.3
                if two:
                    h *= 1.55
                oh = w * 0.09
                top = by - h
                walls_.append([(x, by), (x, top), (x + w, top), (x + w, by)])
                kind = rng.random()
                if kind < 0.5:            # gable end facing us
                    rh = rng.uniform(0.32, 0.45) * w
                    ap = (x + w * 0.5, top - rh)
                    roofs_.append([(x - oh, top + 0.5), (ap[0], ap[1]), (x + w + oh, top + 0.5)])
                    lit_.append([ap, (x + w + oh, top + 0.5), (ap[0], top + 0.5)])
                    edges_.append([(x - oh, top), ap, (x + w + oh, top)])
                elif kind < 0.85:         # hip / side-gable roof (ridge along the street)
                    rh = rng.uniform(0.2, 0.3) * w
                    ins = rng.uniform(0.12, 0.25) * w
                    roofs_.append([(x - oh, top + 0.5), (x + ins, top - rh), (x + w - ins, top - rh),
                                   (x + w + oh, top + 0.5)])
                    lit_.append([(x + w - ins, top - rh), (x + w + oh, top + 0.5), (x + w - ins, top + 0.5)])
                    edges_.append([(x - oh, top), (x + ins, top - rh), (x + w - ins, top - rh), (x + w + oh, top)])
                else:                     # flat-roofed shop / apartment with parapet + water tank
                    ph = 0.0012 * H * scl
                    roofs_.append([(x - 0.5, top + 0.5), (x - 0.5, top - ph), (x + w + 0.5, top - ph),
                                   (x + w + 0.5, top + 0.5)])
                    edges_.append([(x, top - ph), (x + w, top - ph)])
                    if rng.random() < 0.5:
                        tx_ = x + w * rng.uniform(0.55, 0.8)
                        tw_, th2 = 0.0035 * W * scl, 0.004 * H * scl
                        roofs_.append([(tx_, top - ph), (tx_, top - ph - th2), (tx_ + tw_, top - ph - th2),
                                       (tx_ + tw_, top - ph)])
                nwin = max(int(w / (0.0045 * W * scl)), 1)
                floors = 2 if two else 1
                for fl_i in range(floors):
                    for k in range(nwin):
                        if rng.random() < 0.42:
                            wide = rng.random() < 0.3
                            ww = max((0.0036 if wide else 0.0022) * W * scl, 1.0)
                            wh = max(0.0028 * H * scl, 1.0)
                            wx0 = x + (k + 0.5) * w / nwin - ww / 2
                            wy0 = by - (h / floors) * (fl_i + 0.72)
                            col = np.array(win_pal[rng.choice(len(win_pal), p=win_p)])
                            win_rects.append((wx0, wy0, wx0 + ww, wy0 + wh, col, rng.uniform(0.6, 1.8)))
                x += w + 2 * oh + rng.uniform(0.0, 0.008) * W
            house_rows.append((walls_, roofs_, lit_, edges_, ri))
        sx0, sx1 = 0.285 * Wp, 0.285 * Wp + 0.06 * W
        sb = base - 0.012 * H
        shh = 0.026 * H
        polys_back.append([(sx0, sb), (sx0, sb - shh), (sx1, sb - shh), (sx1, sb)])
        ctx = (sx0 + sx1) / 2
        ctw, cth = 0.009 * W, 0.02 * H
        polys_back.append([(ctx - ctw / 2, sb - shh + 1), (ctx - ctw / 2, sb - shh - cth),
                           (ctx - ctw * 0.7, sb - shh - cth), (ctx, sb - shh - cth - 0.009 * H),
                           (ctx + ctw * 0.7, sb - shh - cth), (ctx + ctw / 2, sb - shh - cth),
                           (ctx + ctw / 2, sb - shh + 1)])
        self.clock = (ctx, sb - shh - cth * 0.55)
        for fl_i in range(3):
            for k in range(14):
                if rng.random() < 0.22:
                    wx0 = sx0 + (k + 0.25) * (sx1 - sx0) / 14
                    wy0 = sb - shh * (0.85 - 0.3 * fl_i)
                    win_rects.append((wx0, wy0, wx0 + 0.0026 * W, wy0 + 0.004 * H,
                                      np.array([0.8, 0.93, 1.0]), rng.uniform(0.8, 1.3)))
        fx0 = 0.395 * Wp
        fth = 0.05 * H
        fw = 0.006 * W
        fire_lines = [[(fx0 - fw, base), (fx0 - fw * 0.35, base - fth)], [(fx0 + fw, base), (fx0 + fw * 0.35, base - fth)]]
        for q in (0.25, 0.5, 0.75):
            yq = base - fth * q
            wq = fw * (1 - 0.65 * q)
            fire_lines.append([(fx0 - wq, yq), (fx0 + wq, yq)])
        for q0, q1 in ((0, 0.25), (0.25, 0.5), (0.5, 0.75)):
            a0_, a1_ = fw * (1 - 0.65 * q0), fw * (1 - 0.65 * q1)
            fire_lines.append([(fx0 - a0_, base - fth * q0), (fx0 + a1_, base - fth * q1)])
            fire_lines.append([(fx0 + a0_, base - fth * q0), (fx0 - a1_, base - fth * q1)])
        polys_back.append([(fx0 - fw * 0.9, base - fth), (fx0, base - fth - 0.012 * H), (fx0 + fw * 0.9, base - fth)])
        polys_back.append([(fx0 - fw * 0.5, base - fth), (fx0 - fw * 0.5, base - fth + 0.006 * H),
                           (fx0 + fw * 0.5, base - fth + 0.006 * H), (fx0 + fw * 0.5, base - fth)])
        self.fire_lamp = (fx0, base - fth + 0.003 * H)
        tx_t = 0.452 * Wp
        tor = R.torii_polys(tx_t, y_h - 0.0005 * H, 0.03 * H, 0.03 * W)
        tor_m = P.polys_mask(Wp, Hl, tor)
        poles = []
        px_ = 0.43 * Wp
        while px_ > 0.105 * Wp:
            poles.append((px_, base + 0.001 * H))
            px_ -= rng.uniform(0.028, 0.034) * Wp
        for fxp in (0.085, 0.062, 0.04, 0.018, -0.005):
            xi = int(max(fxp, 0) * Wp * ss)
            poles.append((fxp * Wp, near_top[xi] + 0.004 * H))
        pole_lines, wire_lines = [], []
        ph_ = 0.028 * H
        for (px0, pb) in poles:
            pole_lines.append([(px0, pb), (px0, pb - ph_)])
            pole_lines.append([(px0 - 0.004 * W, pb - ph_ * 0.9), (px0 + 0.004 * W, pb - ph_ * 0.9)])
        for (a, b_) in zip(poles[:-1], poles[1:]):
            for off in (-0.003 * W, 0.003 * W):
                p0 = (a[0] + off, a[1] - ph_ * 0.9)
                p1 = (b_[0] + off, b_[1] - ph_ * 0.9)
                wire_lines.append([tuple(p) for p in C.catenary(p0, p1, 0.0035 * H, 32)])
            wire_lines.append([tuple(p) for p in C.catenary((a[0], a[1] - ph_ * 0.6), (b_[0], b_[1] - ph_ * 0.6),
                                                            0.004 * H, 32)])
        hmask = P.polys_mask(Wp, Hl, polys_back)
        house_col = _hex('#0a1238')
        land = C.over(land, house_col, hmask)
        land_a = np.maximum(land_a, hmask)
        # houses, back row -> front row: plaster walls, dark tiled roofs, comet-lit right slopes, crisp eave rims
        for (walls_, roofs_, lit_, edges_, ri) in house_rows:
            fog = 0.22 * (2 - ri) / 2
            wm = P.polys_mask(Wp, Hl, walls_)
            wc_ = _hex('#121a42') * (1 - fog) + _hex('#223c72') * fog
            land = C.over(land, wc_, wm)
            rm_ = P.polys_mask(Wp, Hl, roofs_)
            rc_ = _hex('#0a1133') * (1 - fog) + _hex('#26437d') * fog
            land = C.over(land, rc_, rm_)
            if lit_:
                lm_ = P.polys_mask(Wp, Hl, lit_) * rm_
                land = C.over(land, _hex('#1d3266') * (1 - fog) + _hex('#34579a') * fog, lm_)
            em_ = P.lines_mask(Wp, Hl, edges_, 0.7 * s + 0.3)
            land = land + em_[..., None] * np.array([0.16, 0.27, 0.46], np.float32)
            land_a = np.maximum(land_a, np.maximum(np.maximum(wm, rm_), em_))
        # shrine hall (irimoya roof with upswept eaves) beside the torii, stone lanterns
        shx, shb = 0.472 * Wp, y_h - 0.0015 * H
        shw, shh = 0.03 * W, 0.009 * H
        sh_walls = [[(shx, shb), (shx, shb - shh), (shx + shw, shb - shh), (shx + shw, shb)]]
        eave = []
        for i in range(17):
            q = i / 16 * 2 - 1
            eave.append((shx + shw / 2 + q * shw * 0.68, shb - shh - 0.004 * H * abs(q) ** 3))
        rtop = shb - shh - 0.014 * H
        sh_roof = [eave[0], (shx + shw * 0.2, rtop + 0.004 * H), (shx + shw * 0.28, rtop), (shx + shw * 0.72, rtop),
                   (shx + shw * 0.8, rtop + 0.004 * H), eave[-1]] + [(p_[0], p_[1] + 0.0025 * H) for p_ in eave[::-1]]
        land = C.over(land, _hex('#1b1f4a'), P.polys_mask(Wp, Hl, sh_walls))
        srm = P.polys_mask(Wp, Hl, [sh_roof])
        land = C.over(land, _hex('#0b1236'), srm)
        sre = P.lines_mask(Wp, Hl, [[eave[0], (shx + shw * 0.2, rtop + 0.004 * H), (shx + shw * 0.28, rtop),
                                     (shx + shw * 0.72, rtop), (shx + shw * 0.8, rtop + 0.004 * H), eave[-1]]],
                           0.7 * s + 0.3)
        land = land + sre[..., None] * np.array([0.2, 0.32, 0.52], np.float32)
        land_a = np.maximum(land_a, np.maximum(srm, P.polys_mask(Wp, Hl, sh_walls)))
        self.shrine = (shx, shb, shw, shh)
        # faint comet-lit roof edges (tops of the house silhouettes)
        hs = np.zeros_like(hmask)
        dd = max(int(round(1.0 * s)), 1)
        hs[dd:] = hmask[:-dd]
        land += np.clip(hmask - hs, 0, 1)[..., None] * np.array([0.12, 0.2, 0.34], np.float32)
        fmask2 = P.lines_mask(Wp, Hl, fire_lines, 0.9 * s + 0.3)
        land = C.over(land, house_col, fmask2)
        land_a = np.maximum(land_a, fmask2)
        land = C.over(land, _hex('#1e0c1e'), tor_m)
        land_a = np.maximum(land_a, tor_m)
        pmask = P.lines_mask(Wp, Hl, pole_lines, 1.1 * s + 0.3)
        wmask = P.lines_mask(Wp, Hl, wire_lines, 0.55 * s + 0.25) * 0.85
        land = C.over(land, _hex('#080f30'), np.maximum(pmask, wmask))
        land_a = np.maximum(land_a, np.maximum(pmask, wmask))

        # lit windows (HDR) with small bloom halos
        wimg = np.zeros((Hl, Wp, 3), np.float32)
        for (x0, y0, x1, y1, col, it) in win_rects:
            xi0, yi0 = int(x0), int(y0)
            xi1, yi1 = max(int(math.ceil(x1)), xi0 + 1), max(int(math.ceil(y1)), yi0 + 1)
            wimg[yi0:yi1, xi0:xi1] += np.asarray(col, np.float32) * it
            lights.append(((x0 + x1) / 2, (y0 + y1) / 2, np.asarray(col, np.float32), it * 0.35, (x1 - x0)))
        land = land + C.blur(wimg, 0.4 * s + 0.2) * 1.0
        halo_w = C.blur(wimg, 3.0 * s + 0.5) * 0.5 + C.blur(wimg, 9 * s + 1) * 0.35
        near_top1 = near_top.reshape(Wp, ss).min(1)
        lamp_posts = []
        lx = 0.115 * Wp
        while lx < 0.78 * Wp:
            ty = near_top1[int(lx)] if lx > 0.44 * Wp else base
            if lx > 0.44 * Wp:
                ty = min(ty, y_h - 0.003 * H)
            sodium = rng.random() < 0.65
            col = np.array([1.0, 0.66, 0.3], np.float32) if sodium else np.array([0.8, 0.92, 1.0], np.float32)
            yl = min(y_h - 0.006 * H, ty)
            if lx < 0.44 * Wp:            # street lamp on a post along the shore road
                yl = base - 0.017 * H
                lamp_posts.append([(lx, base + 0.001 * H), (lx, yl), (lx + 0.0035 * W, yl)])
                lights.append((lx + 0.0035 * W, yl + 0.0008 * H, col, 1.0, 1.5 * s))
            else:
                lights.append((lx, yl, col, 0.9, 1.4 * s))
            lx += rng.uniform(0.02, 0.05) * Wp
        lpm = P.lines_mask(Wp, Hl, lamp_posts, 0.6 * s + 0.3)
        land = C.over(land, _hex('#0a1236'), lpm)
        land_a = np.maximum(land_a, lpm)
        shx, shb, shw, shh = self.shrine
        for q_ in (0.15, 0.85):           # shrine lanterns
            lights.append((shx + shw * q_, shb - shh * 0.45, np.array([1.0, 0.62, 0.3], np.float32), 0.9, 1.3 * s))
        for fx_ in (0.03, 0.06, 0.53, 0.66, 0.74, 0.93):
            xi = int(fx_ * Wp)
            yl = min(near_top1[xi] + rng.uniform(0.006, 0.014) * H, y_h - 0.002 * H)
            lights.append((fx_ * Wp, yl, np.array([1.0, 0.75, 0.42], np.float32), 0.6, 1.0 * s))
        lights.append((self.clock[0], self.clock[1], np.array([1.0, 0.92, 0.75], np.float32), 1.2, 2.2 * s))
        lights.append((self.fire_lamp[0], self.fire_lamp[1], np.array([1.0, 0.55, 0.25], np.float32), 1.0, 1.4 * s))
        lights.append((tx_t - 0.022 * W, y_h - 0.006 * H, np.array([1.0, 0.7, 0.35], np.float32), 1.0, 1.3 * s))
        lights.append((tx_t + 0.022 * W, y_h - 0.006 * H, np.array([1.0, 0.7, 0.35], np.float32), 1.0, 1.3 * s))
        limg = np.zeros((Hl, Wp, 3), np.float32)
        for (x0, y0, col, it, r) in lights:
            if r > 2.5 * s:
                C.splat(limg, x0, y0, 2.2 * s, col, it * 1.5)
                C.splat(limg, x0, y0, 9 * s, col, it * 0.1)
                continue
            C.splat(limg, x0, y0, max(r, 0.8), col, it * 2.0)
            C.splat(limg, x0, y0, 6 * s, col, it * 0.14)
        xl, yl_ = C.grid(Wp, Hl)
        th_ = np.exp(-((xl - 0.27 * Wp) / (0.13 * Wp)) ** 2) * np.exp(-np.clip(base - yl_, 0, None) / (0.025 * H)) * \
            (yl_ < y_h)
        limg += th_[..., None] * np.array([0.16, 0.08, 0.03], np.float32)
        # glow over the sky just above the roofs is composited additively (trucks with the town)
        glow = limg + halo_w
        land = land + glow * land_a[..., None]
        self.town_glow = np.ascontiguousarray((glow * (1 - land_a[..., None]))[b0:Hl]).astype(np.float32)
        town = np.dstack([land, land_a]).astype(np.float32)
        layers.append((town, 1.0))
        self.lights = lights
        del wimg, halo_w, limg, xl, yl_, th_, glow

        self.layers = [(np.ascontiguousarray(l[b0:Hl]).astype(np.float32), f) for (l, f) in layers]
        # half-res copies for the per-frame reflection band
        self.layers_h = [(cv2.resize(l, (Wp // 2, (Hl - b0) // 2), interpolation=cv2.INTER_AREA), f)
                         for (l, f) in self.layers]
        self.glow_h = cv2.resize(self.town_glow, (Wp // 2, (Hl - b0) // 2), interpolation=cv2.INTER_AREA)

        # ------------------------------------------------------------------ reflection plate
        full = self.sky + self.comet_static + full_tail
        refl = cv2.GaussianBlur(full, (0, 0), sigmaX=0.8 * s + 0.2, sigmaY=1.8 * s + 0.3)
        soft = C.blur(self.comet_static + full_tail, 7 * s + 0.5) * 0.45
        refl += soft
        stw = self.stars.copy()
        glint = cv2.GaussianBlur(stw, (0, 0), sigmaX=2.6 * s + 0.3, sigmaY=0.5 * s + 0.2) * 2.2
        tint = np.array([0.8, 0.86, 0.95], np.float32)
        self.refl_tint = tint
        # sky-only band (half res) and the additive glints/soft glow of the band (full res)
        rb = cv2.GaussianBlur(full[b0:Hl], (0, 0), sigmaX=0.8 * s + 0.2, sigmaY=1.8 * s + 0.3)
        self.rb_sky_h = cv2.resize(rb, (Wp // 2, (Hl - b0) // 2), interpolation=cv2.INTER_AREA)
        self.rb_glint = np.ascontiguousarray(glint[b0:Hl]).astype(np.float32)
        self.rb_soft = np.ascontiguousarray(soft[b0:Hl]).astype(np.float32)
        refl += glint
        self.refl = (refl * tint).astype(np.float32)
        del full, stw, glint, soft, full_tail, rb

        # light reflection streak plates (lake rows only): town lights (truck with the town) + comet head
        Hlk = Hp - int(y_h) + 2
        self.lk0 = int(y_h)

        def streaks(src):
            lk = np.zeros((Hlk, Wp, 3), np.float32)
            for (x0, y0, col, it, r) in src:
                ym = 2 * y_h - y0 - self.lk0
                Ls = (0.05 + 0.02 * it) * H
                wdt = max(min(r, 3 * s), 1.0) * 1.2 + 1.5 * s
                x0i, x1i = int(max(x0 - 5 * wdt, 0)), int(min(x0 + 5 * wdt + 1, Wp))
                y0i, y1i = int(max(ym - 2, 0)), int(min(ym + Ls * 3, Hlk))
                if x1i <= x0i or y1i <= y0i:
                    continue
                yy2, xx2 = np.mgrid[y0i:y1i, x0i:x1i].astype(np.float32)
                dyy = yy2 - ym
                wd = wdt * (1 + dyy / (0.1 * H))
                k = np.exp(-((xx2 - x0) / wd) ** 2) * np.exp(-np.clip(dyy, 0, None) / Ls) * C.smoothstep(-2, 1, dyy)
                lk[y0i:y1i, x0i:x1i] += k[..., None] * np.asarray(col, np.float32) * it * 0.8
            return lk
        lk = HL.light_streaks(Hlk, Wp, lights, y_h, self.lk0, H, s, seed=5)
        xk, yk = C.grid(Wp, Hlk)
        spill = np.exp(-((xk - 0.27 * Wp) / (0.12 * Wp)) ** 2) * np.exp(-yk / (0.035 * H))
        lk += spill[..., None] * np.array([0.22, 0.11, 0.04], np.float32)
        self.lk = lk
        # comet glitter path: from the far shore straight down past the mirrored nucleus (with the tail's
        # cool glow broadened around it) - broken up by ripples per frame
        self.lkc = V4.glitter_column(Hlk, Wp, hx, 0.0, 2 * y_h - hy - self.lk0, H, s,
                                     (0.5, 0.78, 1.0), (0.05, 0.12, 0.24))
        self.lkc += streaks([(hx, hy, np.array([0.6, 0.85, 1.0]), 1.6, 8 * s)])
        self.lkc += V4.tail_glitter(Hlk, Wp, self.head, d1, L, y_h, self.lk0, H, s, (0.35, 0.62, 1.0), amt=0.5,
                                    bend=0.2)
        del xk, yk, spill

        # ------------------------------------------------------------------ water textures
        self.T1 = P.tile_noise(256, 41, 4.0, 20.0)
        self.T2 = P.tile_noise(256, 43, 7.0, 30.0)
        self.T3 = P.tile_noise(256, 45, 10.0, 10.0)
        wy = np.arange(Hlk, dtype=np.float32) / (0.5 * H)
        wc0, wc1 = _hex('#1c2c66'), _hex('#070f34')
        self.water_grad = (wc0 + (wc1 - wc0) * np.clip(wy, 0, 1)[:, None] ** 0.6).astype(np.float32)

        # ------------------------------------------------------------------ mist (two low bands, lit by comet)
        from lib import clouds as K
        mw_ = int(Wp * 1.3)
        mh = int(0.12 * H) + 2
        stk = K.streaks(mw_, mh, 62, xcells=4.0, ycells=10.0, octaves=4)
        stk2 = K.streaks(mw_, mh, 63, xcells=3.0, ycells=8.0, octaves=4)
        yv = np.arange(mh, dtype=np.float32)[:, None] / mh
        prof1 = np.exp(-((yv - 0.62) / 0.14) ** 2)
        prof2 = np.exp(-((yv - 0.38) / 0.12) ** 2)
        a1m = C.smoothstep(0.35, 0.85, stk) * prof1 * 0.4
        a2m = C.smoothstep(0.45, 0.9, stk2) * prof2 * 0.28
        self.mist_a = np.clip(a1m + a2m, 0, 0.6).astype(np.float32)
        self.mist_y0 = y_h - 0.075 * H
        mx = np.arange(mw_, dtype=np.float32) - (mw_ - Wp) / 2
        lit = np.exp(-((mx - hx) / (0.3 * Wp)) ** 2)
        warm = np.exp(-((mx - 0.27 * Wp) / (0.12 * Wp)) ** 2)
        mc = (np.array([0.32, 0.46, 0.78], np.float32)[None, :] + lit[:, None] * np.array([0.2, 0.3, 0.35]) +
              warm[:, None] * np.array([0.25, 0.12, 0.02]))
        self.mist_col = np.repeat(mc[None, :, :], mh, 0).astype(np.float32)

        # ------------------------------------------------------------------ foreground reeds
        rr = np.random.default_rng(9)
        blades = []
        for side, (a0, a1_, lean) in enumerate(((-0.08, 0.2, 0.25), (0.83, 1.08, -0.2))):
            for i in range(80):
                bx = rr.uniform(a0, a1_)
                edge = (1 - abs((bx - (a0 if side == 0 else a1_)) / (a1_ - a0)))
                h = rr.uniform(0.06, 0.26) * (0.4 + 0.8 * edge)
                bl = lean * rr.uniform(0.2, 1.2) + rr.normal(0, 0.1)
                blades.append((bx, h, bl, rr.uniform(0.0025, 0.0045), rr.uniform(0, 6.28), rr.uniform(0.9, 1.6),
                               rr.random() < 0.12))
        self.blades = blades

        self.meteors = [(0.55, 0.45, 0.2, my + 0.14 * H, 152, 0.14, 1.0),
                        (2.35, 0.35, 0.86, my + 0.42 * H, 140, 0.11, 0.7),
                        (3.9, 0.5, 0.3, my + 0.6 * H, 158, 0.13, 1.0)]
        self.xs_f, self.ys_f = C.grid(W, H)

    # ---------------------------------------------------------------------------------------------
    def cam(self, t):
        u = t / DURATION
        e = 0.12 * u + 0.88 * C.ease_in_out_sine(u)
        z = 1.0 + 0.03 * u
        cy = self.my + self.H / 2 + self.tilt * e
        tr = self.truck * (0.2 * u + 0.8 * C.ease_in_out_sine(u) - 0.5)
        return cy, z, e, tr

    def _M(self, cy, z):
        W, H = self.W, self.H
        return np.array([[1 / z, 0, self.Wp / 2 - W / 2 / z], [0, 1 / z, cy - H / 2 / z]], np.float32)

    def _warp_rows(self, plate, M, r1, border=cv2.BORDER_CONSTANT):
        if r1 <= 0:
            return None
        return cv2.warpAffine(plate, M, (self.W, r1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=border)

    def _warp_band(self, plate, M, dx, y_off, fr0, fr1, border=cv2.BORDER_CONSTANT):
        """Warp a band plate (rows start at plate y = y_off), shifted right by dx, into frame rows [fr0, fr1)."""
        z = 1.0 / M[0, 0]
        Mb = M.copy()
        Mb[0, 2] -= dx
        Mb[1, 2] = M[1, 2] - y_off + fr0 / z
        return cv2.warpAffine(plate, Mb, (self.W, fr1 - fr0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=border)

    def frame(self, t):
        W, H, s = self.W, self.H, self.s
        cy, z, e, tr = self.cam(t)
        M = self._M(cy, z)
        yh_f = (self.y_h - M[1, 2]) * z
        r0 = int(min(max(math.floor(yh_f), 0), H))       # first lake row
        rs = min(r0 + 2, H)                                # sky/land rows to render
        u = t / DURATION

        img = np.zeros((H, W, 3), np.float32)
        met = np.zeros((H, W, 3), np.float32)
        if rs > 0:
            sk = self._warp_rows(self.sky, M, rs, cv2.BORDER_REPLICATE)
            stv = self._warp_rows(self.stars, M, rs)
            M2 = M.copy()
            M2[:, :2] *= 2
            hr = (rs + 1) // 2
            ph = cv2.warpAffine(self.phase, M2, (W // 2, hr), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)
            tw = 0.8 + 0.2 * np.sin(t * (1.6 + 2.4 * ph) + ph * 40.0)
            tw = cv2.resize(tw, (W, hr * 2), interpolation=cv2.INTER_NEAREST)[:rs]
            sk += stv * tw[..., None]
            sk += self._warp_rows(self.comet_static, M, rs)
            # tail: slowly lengthening, with gently flowing streamers
            tx0, ty0 = self.tbox
            th_, tw_ = self.tail.shape[:2]
            fx0 = int(max((tx0 - M[0, 2]) * z - 2, 0)); fx1 = int(min((tx0 + tw_ - M[0, 2]) * z + 2, W))
            fy0 = int(max((ty0 - M[1, 2]) * z - 2, 0)); fy1 = int(min((ty0 + th_ - M[1, 2]) * z + 2, rs))
            if fx1 > fx0 and fy1 > fy0:
                Mt = np.array([[1 / z, 0, M[0, 2] - tx0 + fx0 / z], [0, 1 / z, M[1, 2] - ty0 + fy0 / z]], np.float32)
                tl = cv2.warpAffine(self.tail, Mt, (fx1 - fx0, fy1 - fy0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT)
                aux = cv2.warpAffine(self.tail_aux, Mt, (fx1 - fx0, fy1 - fy0),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
                uu, vn, bd = aux[..., 0], aux[..., 1], aux[..., 2]
                Lt = 0.64 + 0.4 * (0.25 * u + 0.75 * C.ease_in_out_sine(u))
                grow = 1 - C.smoothstep(Lt - 0.28, Lt, uu)
                # streamers flow outward along the tail (knots travel away from the nucleus)
                flow = (np.sin(vn * 9.0 + uu * 22.0 - t * 2.2) * 0.6 + np.sin(vn * 19.0 - uu * 13.0 - t * 1.5) * 0.4)
                tl = tl * grow[..., None] + (bd * flow * grow)[..., None] * np.array([0.3, 0.5, 0.8], np.float32) * 0.22
                sk[fy0:fy1, fx0:fx1] += tl
            # fragment sparkle (slow, smooth glint)
            nsp = len(self.sparks)
            for i, (x, y, it, col) in enumerate(self.sparks):
                fxs, fys = (x - M[0, 2]) * z, (y - M[1, 2]) * z
                if 0 <= fys < rs:
                    k = 0.72 + 0.28 * math.sin(t * (1.3 + 0.37 * i) + i * 2.1)
                    C.splat(sk, fxs, fys, 0.9 * s + 0.35, np.asarray(col, np.float32), 2.0 * it * k)
                    C.splat(sk, fxs, fys, 4.0 * s + 0.5, np.asarray(col, np.float32), 0.12 * it * k)
                    if i >= nsp - self.n_front:
                        C.splat(sk, fxs, fys, 1.3 * s + 0.4, np.asarray(col, np.float32), 2.2 * it * k)
                        V4.cross_flare(sk, fxs, fys, (12 + 7 * it) * s, 0.5 * s + 0.3, col, 0.8 * it * k, rot=0.3)
            # hot nucleus: short cross-flare
            hfx0, hfy0 = (self.head[0] - M[0, 2]) * z, (self.head[1] - M[1, 2]) * z
            if -40 < hfy0 < rs + 40:
                V4.cross_flare(sk, hfx0, hfy0, 34 * s, 0.7 * s + 0.3, (0.8, 0.93, 1.0), 0.9, rot=0.3)
                V4.cross_flare(sk, hfx0, hfy0, 16 * s, 0.6 * s + 0.3, (0.9, 0.97, 1.0), 0.5, rot=0.3 + math.pi / 4)
            for (t0, dur, x0f, y0, angd, lenf, br) in self.meteors:
                qq = (t - t0) / dur
                if qq < 0 or qq > 1.25:
                    continue
                a = math.radians(angd)
                dx, dy = math.cos(a), math.sin(a)
                travel = lenf * W * 1.3
                hx = x0f * self.Wp + dx * travel * min(qq, 1.0)
                hy = y0 + dy * travel * min(qq, 1.0)
                tail = lenf * W * min(qq * 1.6, 1.0)
                env = math.sin(min(qq, 1.0) * math.pi) ** 0.7 if qq <= 1 else 0.0
                fade_after = max(0.0, 1 - (qq - 1) / 0.25) if qq > 1 else 0.0
                self._streak(met, (hx - M[0, 2]) * z, (hy - M[1, 2]) * z, -dx, -dy, tail * z, 1.1 * s + 0.4,
                             br * max(env, fade_after * 0.25))
            sk += met[:rs]
            img[:rs] = sk
            # land planes with parallax
            fr0 = int(max(math.floor((self.b0 - M[1, 2]) * z), 0))
            if rs > fr0:
                band = img[fr0:rs]
                for k_, (lay, f) in enumerate(self.layers):
                    dft = self.cloud_drift * t if k_ == self.cloud_layer else 0.0
                    lw = self._warp_band(lay, M, f * tr + dft, self.b0, fr0, rs)
                    band = F.over_rgba(band, lw)
                band += self._warp_band(self.town_glow, M, tr, self.b0, fr0, rs)
                img[fr0:rs] = band

        blink = max(0.0, math.sin(t * 2 * math.pi * 0.6)) ** 3
        fm = 0.55 * tr
        tx = (self.tower_light[0] + fm - M[0, 2]) * z
        ty = (self.tower_light[1] - M[1, 2]) * z
        C.splat(img, tx, ty, 1.4 * s + 0.3, np.array([1.0, 0.15, 0.1], np.float32), 2.5 * blink + 0.15)
        C.splat(img, tx, ty, 8 * s, np.array([1.0, 0.2, 0.1], np.float32), 0.12 * blink)
        cxp = (0.47 + 0.05 * t / DURATION) * self.Wp + tr
        cyp = self.y_h - 0.005 * H
        carx = (cxp - M[0, 2]) * z
        cary = (cyp - M[1, 2]) * z
        C.splat(img, carx, cary, 1.3 * s + 0.3, np.array([1.0, 0.95, 0.85], np.float32), 2.0)
        C.splat(img, carx, cary, 7 * s, np.array([1.0, 0.9, 0.7], np.float32), 0.12)

        # ---- lake
        if r0 < H:
            self._t = t
            self._update_refl_band(tr)
            px = self.xs_f[r0:] / z + M[0, 2]
            py = self.ys_f[r0:] / z + M[1, 2]
            img[r0:] = self._lake(t, px, py, met, M, z, (carx, cary), yh_f, r0, tr)

        # ---- mist over the far shore / waterline (drifts, trucks with the shore)
        mo = t * 0.004 * W - 0.8 * tr
        mh, mw_ = self.mist_a.shape
        Mm = np.array([[1 / z, 0, M[0, 2] + (mw_ - self.Wp) / 2 + mo], [0, 1 / z, M[1, 2] - self.mist_y0]], np.float32)
        my0 = int(max((self.mist_y0 - M[1, 2]) * z - 2, 0))
        my1 = int(min((self.mist_y0 + mh - M[1, 2]) * z + 2, H))
        if my1 > my0:
            Mm2 = Mm.copy()
            Mm2[1, 2] += my0 / z
            ma = cv2.warpAffine(self.mist_a, Mm2, (W, my1 - my0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)[..., None]
            mc = cv2.warpAffine(self.mist_col, Mm2, (W, my1 - my0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
            img[my0:my1] = img[my0:my1] * (1 - ma) + mc * ma

        # ---- lens-flare ghosts from the comet head
        hfx = (self.head[0] - M[0, 2]) * z
        hfy = (self.head[1] - M[1, 2]) * z
        if -0.2 * H < hfy < 1.1 * H:
            vx, vy = W / 2 - hfx, H / 2 - hfy
            for kk, rad, col, it in ((0.55, 0.012, (0.4, 0.7, 1.0), 0.04), (1.25, 0.03, (0.5, 0.9, 0.8), 0.025),
                                     (1.7, 0.018, (0.8, 0.6, 1.0), 0.03)):
                C.splat(img, hfx + vx * kk, hfy + vy * kk, rad * W, np.array(col, np.float32), it)

        img = self._reeds(img, t, e, z, 2.0 * tr)

        img = self._bloom(img)
        img = F.shoulder(img, 0.85, desat=0.2)
        return F.finish_fast(img, t, sat=1.08, grain_amt=0.003, vig=0.3, ca=0.0006)

    # ---------------------------------------------------------------------------------------------
    def _update_refl_band(self, tr):
        """Rebuild the reflection of the land band with this frame's parallax offsets (half res)."""
        s = self.s
        comp = self.rb_sky_h.copy()
        hh, ww = comp.shape[:2]
        alpha = np.zeros((hh, ww), np.float32)
        for k_, (lay, f) in enumerate(self.layers_h):
            dft = self.cloud_drift * self._t if k_ == self.cloud_layer else 0.0
            Mh = np.array([[1, 0, -(f * tr + dft) / 2], [0, 1, 0]], np.float32)
            lw = cv2.warpAffine(lay, Mh, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            comp = F.over_rgba(comp, lw)
            alpha = np.maximum(alpha, lw[..., 3])
        Mh = np.array([[1, 0, -tr / 2], [0, 1, 0]], np.float32)
        comp += cv2.warpAffine(self.glow_h, Mh, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        comp = cv2.GaussianBlur(comp, (0, 0), sigmaX=0.4 * s + 0.1, sigmaY=0.9 * s + 0.15)
        Hb = self.Hl - self.b0
        big = cv2.resize(comp, (self.Wp, Hb), interpolation=cv2.INTER_LINEAR)
        ab = cv2.resize(alpha, (self.Wp, Hb), interpolation=cv2.INTER_LINEAR)
        big += self.rb_soft + self.rb_glint * (1 - ab[..., None])
        self.refl[self.b0:self.Hl] = big * self.refl_tint

    def _bloom(self, img, threshold=0.75, knee=0.3, strength=0.35):
        H, W = self.H, self.W
        q = 4
        small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
        lum = small.max(-1, keepdims=True)
        k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1) ** 2
        br = small * k
        acc = np.zeros_like(br)
        for r, wt in ((0.004, 1.0), (0.012, 0.8), (0.035, 0.6), (0.09, 0.45)):
            acc += F.fast_blur(br, max(r * W / q, 0.6)) * wt
        acc /= 2.85
        return img + cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR) * strength

    def _streak(self, img, hx, hy, dx, dy, length, width, br):
        if br <= 0.001:
            return
        H, W = img.shape[:2]
        x1, y1 = hx + dx * length, hy + dy * length
        pad = 6 * width + 4
        bx0, bx1 = int(max(min(hx, x1) - pad, 0)), int(min(max(hx, x1) + pad, W))
        by0, by1 = int(max(min(hy, y1) - pad, 0)), int(min(max(hy, y1) + pad, H))
        if bx1 <= bx0 or by1 <= by0:
            return
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        rx, ry = xx - hx, yy - hy
        u = (rx * dx + ry * dy) / max(length, 1)
        v = -rx * dy + ry * dx
        prof = C.smoothstep(-0.02, 0.02, u) * np.clip(1 - u, 0, 1) ** 2
        core = np.exp(-(v / width) ** 2) * prof
        glow = np.exp(-(v / (width * 4)) ** 2) * prof * 0.15
        head = np.exp(-(rx * rx + ry * ry) / (width * 2.5) ** 2) * 1.5
        k = (core + glow + head) * br
        img[by0:by1, bx0:bx1] += k[..., None] * np.array([0.8, 0.92, 1.0], np.float32) * 1.6

    def _lake(self, t, px, py, met, M, z, car, yh_f, r0, tr):
        H, W, s = self.H, self.W, self.s
        y_h = self.y_h
        dyw = np.maximum(py - y_h, 0.25)
        D = (0.05 * H) / dyw
        X = (px - tr - self.Wp / 2) / dyw
        N = 256.0
        tu1 = X * 60.0 + t * 1.5
        tv1 = D * 700.0 - t * 5.0
        tu2 = X * 110.0 - t * 2.0
        tv2 = D * 1100.0 - t * 7.0
        n1 = cv2.remap(self.T1, np.mod(tu1, N).astype(np.float32), np.mod(tv1, N).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n2 = cv2.remap(self.T2, np.mod(tu2, N).astype(np.float32), np.mod(tv2, N).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n3 = cv2.remap(self.T3, np.mod(tu1 * 0.7 + 50, N).astype(np.float32), np.mod(tv1 * 0.5 + 90, N).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dDdy = (0.05 * H) / (dyw * dyw) / z
        aa = np.clip(1.5 / np.maximum(dDdy * 700.0, 1e-3), 0, 1)
        nn = (n1 * 0.65 + n2 * 0.35) * aa
        A = 0.02 * dyw + 0.4 * s
        ph1 = D * 600.0 + n1 * 2.2 + X * 1.5 - t * 1.6
        ph2 = D * 1100.0 - n2 * 1.8 - X * 2.5 - t * 2.3
        aa1 = np.clip(1.6 - dDdy * 600.0 * 1.2, 0, 1)
        aa2 = np.clip(1.6 - dDdy * 1100.0 * 1.2, 0, 1)
        env_ = np.clip(0.55 + 0.45 * n3, 0.1, 1.0)
        sw = (np.sin(ph1) * aa1 * 0.6 + np.sin(ph2) * aa2 * 0.4) * env_
        sx = px + n3 * aa * A * 0.3 + sw * A * 0.7
        sy = 2 * y_h - py + nn * A
        sy = np.clip(sy, 0, self.Hp - 1)
        refl = cv2.remap(self.refl, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        fres = 0.66 + 0.26 * np.exp(-dyw / (0.14 * H))
        rows = np.clip((py[:, 0] - self.lk0).astype(np.int32), 0, self.water_grad.shape[0] - 1)
        wc = self.water_grad[rows][:, None, :]
        mod = 1 + 0.12 * nn / 2.5 * aa
        out = wc * (1 - fres[..., None]) + refl * (fres * mod)[..., None]
        # broken light columns (town lights truck with the town; the comet streak stays with the sky)
        lky = (py - self.lk0 + nn * A * 0.3).astype(np.float32)
        lkx = px + n3 * aa * A * 0.8 + sw * A * 0.6
        lk = cv2.remap(self.lk, (lkx - tr).astype(np.float32), lky, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        nb = cv2.remap(self.T2, np.mod(tu2 * 1.7 + 30, N).astype(np.float32), np.mod(tv2 * 2.3 + 70, N).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        band = np.clip(0.55 + 0.45 * n2 * aa + 0.4 * nb * aa, 0.05, 1.4)
        out += lk * band[..., None]
        # comet glitter: ripple crests catch the light -> broken horizontal dashes along the column
        lc = cv2.remap(self.lkc, (lkx + sw * A * 0.8).astype(np.float32), lky, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        crest = np.clip(0.5 + 0.5 * np.sin(ph1 * 1.0 + n2 * 1.5) * aa1 + 0.35 * nb * aa, 0, 1.2)
        glit = 0.25 + 1.35 * C.smoothstep(0.45, 0.95, crest)
        out += lc * glit[..., None]
        sheen = np.exp(-dyw / (0.012 * H))
        out += sheen[..., None] * np.array([0.08, 0.14, 0.22], np.float32)
        if met.any():
            rr = np.arange(r0, H, dtype=np.float32)
            src_rows = np.clip(2 * yh_f - rr, 0, H - 1).astype(np.float32)
            mx = np.repeat(np.arange(W, dtype=np.float32)[None, :], len(rr), 0) + n3 * aa * A * 0.5
            my_ = np.repeat(src_rows[:, None], W, 1) + nn * A
            mr = cv2.remap(met, mx.astype(np.float32), my_.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
            out += C.blur(mr, 1.5 * s) * 0.5
        cx, cyy = car
        ym = 2 * yh_f - cyy - r0
        if 0 <= ym < px.shape[0]:
            yy = np.arange(px.shape[0], dtype=np.float32)[:, None] - ym
            xx = np.arange(W, dtype=np.float32)[None, :] - cx
            k = np.exp(-(xx / (2.5 * s + 1)) ** 2) * np.exp(-np.clip(yy, 0, None) / (0.05 * H)) * (yy > -2)
            out += (k * band)[..., None] * np.array([1.0, 0.9, 0.7], np.float32) * 0.6
        return out.astype(np.float32)

    def _reeds(self, img, t, e, z, dxr):
        W, H, s = self.W, self.H, self.s
        base_y = H * 1.01 + (1 - e) * self.tilt * 1.35 * z
        if base_y - 0.3 * H > H:
            return img
        ss = 2
        y0 = int(max(base_y - 0.3 * H, 0))
        if y0 >= H:
            return img
        hh = H - y0
        m = np.zeros((hh * ss, W * ss), np.uint8)
        polys = []
        for (bx, h, lean, wdt, ph, fr, head) in self.blades:
            sway = 0.035 * math.sin(t * fr + ph) + 0.015 * math.sin(t * fr * 2.3 + ph * 1.7)
            x0 = bx * W + dxr
            L = h * H
            pts_l, pts_r = [], []
            n = 7
            for i in range(n + 1):
                q = i / n
                bend = (lean + sway) * q * q
                x = x0 + bend * L * 0.6
                y = base_y - L * q * (1 - 0.15 * abs(lean) * q)
                w = wdt * W * (1 - q) ** 0.9 * 0.5 + 0.3 * s
                pts_l.append((x - w, y))
                pts_r.append((x + w, y))
            poly = pts_l + pts_r[::-1]
            polys.append(np.array([(p[0] * ss, (p[1] - y0) * ss) for p in poly], np.int32))
            if head:
                hx = x0 + (lean + sway) * L * 0.6
                hy = base_y - L * (1 - 0.15 * abs(lean))
                cv2.ellipse(m, (int(hx * ss), int((hy - y0 + 0.02 * H) * ss)),
                            (int(0.0035 * W * ss), int(0.022 * H * ss)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(m, polys, 255, lineType=cv2.LINE_AA)
        a = cv2.resize(m.astype(np.float32) / 255, (W, hh), interpolation=cv2.INTER_AREA)
        d = max(int(round(1.6 * s)), 1)
        sh_r = np.zeros_like(a)
        sh_r[d:, :-d] = a[:-d, d:]
        sh_l = np.zeros_like(a)
        sh_l[d:, d:] = a[:-d, :-d]
        left = (np.arange(W) < W / 2)[None, :]
        rim = np.clip(a - np.where(left, sh_r, sh_l), 0, 1)
        col = np.array([0.012, 0.02, 0.06], np.float32)
        rc = np.array([0.25, 0.42, 0.7], np.float32)
        img[y0:] = img[y0:] * (1 - a[..., None]) + col * a[..., None] + rim[..., None] * rc * 0.55
        return img
