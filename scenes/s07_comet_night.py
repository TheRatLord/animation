"""Round 20 (judges: popcorn/cotton-puff clouds, drawn-on twilight rims, all-cool tail, felt hills, box town,
rigid tilt): clouds rebuilt as painted yn_05 CLUSTERS (s07_comet_night_cloud20): lobed clumps + contour florets
(cauliflower silhouettes), star-filled gaps, satellites, full-res AA silhouettes (crisp tops, lumpy lighter
undersides), a broad moonlit plane with a painted cauliflower light/shadow boundary + soft shadow dabs; the tail's
light streams over them.  Twilight rows are flat-based with crisp violet cauliflower tops and a WIDE graded
peach -> pink rim that fades along the row away from the glow.  Two cloud plates (near/far) with their own
parallax, wind drift and truck response.  Dust tail: warm peach/gold outer half near the nucleus vs the cyan ion
tail, stronger bend.  Camera: blended beta/smootherstep ease + constant crab; reeds move more.  Forest ridges:
treeline detail only where the crest catches sky light, darker lost edges toward the lake (ridge20).  Town:
varied sizes / roof pitches / heights, unlit houses, warm spill at the shore.

Round 19 (reviewer: clay clouds, neon twilight rims, flat rainbow fan, scanline lake, rigid tilt):
soft painted clouds, fewer/bigger, cumulus -> cirrus, moonlit plane only on the top edge, flat body, 20-40 px lost
undersides; twilight banks rim-lit only on the horizon side, fading inward orange -> pink -> violet, hazed
(s07_comet_night_cloud19 + layout19). True split comet: narrow straight cool ion tail + wider curved warm dust
tail (magenta -> pink -> gold) with dark sky between, spectral head fringe (comet19). Lake: stroke texture
confined to wind patches, perspective swell shading, reflections blurred + darkened toward the camera.
Camera: beta(2,3) ease (bulk early, long settle); stars/MW 0.42, comet 0.3, clouds 0.1 lag. Mountains:
facets melted into painted masses, warm sunward wash, violet away side, hazier soft far ranges (mtn19).

Round 15 (final-panel notes): night clouds repainted as flat moonlit yn_05 shapes (s07_comet_night_cloud15):
no bright outline and no dark base band / drop shadow; a soft moonlit gradient (lighter cyan-grey top, darker
blue-violet body, bluer translucent base that lets the stars through), soft per-puff planes, painted dabs,
crisp lit top edges with lost frayed undersides, wind-torn fragments instead of bead chains, warm afterglow on
the undersides of the horizon clouds. Motion: clouds drift independently (0.009 W/s) during the tilt; the lake
reflection trails the land by a few px while the camera moves (_refl_lag) and now mirrors the clouds too.
Round 14: night clouds rebuilt as yn_05 hard-edged cut-outs (s07_comet_night_cloud14 + layout14): unions of
circles at 3 scales, 1px AA, flat moonlit top cel, flat blue-grey body, hard darker base band, inner lobe
separations, detached scraps; no blur/noise erosion/skirt.
s07_comet_night - Night sky over a mountain lake (Your Name homage).

Round 13: clouds repainted from one continuous density field (s07_comet_night_cloud13 + layout13): clustered
broken banks + torn diagonal streamers along the comet, crisp moonlit tops, feathered inner gradient, flat
body, translucent lost undersides, no base band / lip. Range plates flattened into painted planes with aerial
perspective on the back ranges; forests as near-flat masses with a treeline fringe (s07_comet_night_flat13).
Milky Way: violet/brown/magenta dust tint variation, more stars along lane + band edges. Lake: broken ripple
dash strokes, stronger light pillars, darker toward camera + shoreline shadow.

Round 12 (final-panel notes): night clouds repainted as flat yn_05 cut-outs - one moonlit top value with a
crisp scalloped edge, one flat blue-grey body, a hard darker base band with scalloped top, no per-lobe
shading and no smoke under the bases (s07_comet_night_cloud12); Milky Way dust lanes are broad, soft,
feathered violet-brown dust clouds with star density rising along their margins (s07_comet_night_mw12).

Round 8: split comet (broad cyan dust fan + separate magenta / gold outer bands + straight ion tail at
12 deg, s07_comet_night_v6), structured Milky Way (s07_comet_night_mw6), wide gold->salmon->violet
twilight, rock strata / couloirs / notched crests + HDR comet rim with halation and lost shadow edges
(terrain), stacked conifer-top bands with snow-dusted crowns (hills + v6), horizontally sliced lake.

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
import s07_comet_night_v6 as V6  # noqa: E402
import s07_comet_night_mw6 as MW6  # noqa: E402
import s07_comet_night_v7 as V7  # noqa: E402
import s07_comet_night_mw7 as MW7  # noqa: E402
import s07_comet_night_v8 as V8  # noqa: E402
import s07_comet_night_comet8 as CM  # noqa: E402
import s07_comet_night_ridge8 as RL  # noqa: E402
import s07_comet_night_mw8 as MW8  # noqa: E402,F401
import s07_comet_night_mw9 as MW9  # noqa: E402
import s07_comet_night_comet9 as CM9  # noqa: E402
import s07_comet_night_cloud9 as K9  # noqa: E402,F401
import s07_comet_night_cloud12 as K12  # noqa: E402
import s07_comet_night_mw12 as MW12  # noqa: E402
import s07_comet_night_cloud13 as K13  # noqa: E402
import s07_comet_night_cloud14 as K14  # noqa: E402,F401
import s07_comet_night_cloud15 as K15  # noqa: E402
import s07_comet_night_flat13 as FT  # noqa: E402
import s07_comet_night_cloud16 as K16  # noqa: E402,F401
import s07_comet_night_layout16 as LY16  # noqa: E402,F401
import s07_comet_night_cloud17 as K17  # noqa: E402
import s07_comet_night_layout17 as LY17  # noqa: E402,F401
import s07_comet_night_cloud18 as K18  # noqa: E402
import s07_comet_night_layout18 as LY18  # noqa: E402,F401
import s07_comet_night_cloud19 as K19  # noqa: E402
import s07_comet_night_layout19 as LY19  # noqa: E402
import s07_comet_night_comet18 as CM18  # noqa: E402,F401
import s07_comet_night_comet19 as CM19  # noqa: E402
import s07_comet_night_mtn19 as MT19  # noqa: E402
import s07_comet_night_cloud20 as K20  # noqa: E402
import s07_comet_night_cloud21 as K21  # noqa: E402
import s07_comet_night_ridge21 as RG21  # noqa: E402
import s07_comet_night_fg21 as FG21  # noqa: E402
import s07_comet_night_ridge20 as RP20  # noqa: E402
import s07_comet_night_ridge18 as RG18  # noqa: E402
import s07_comet_night_ridge16 as RL16  # noqa: E402
import s07_comet_night_layout13 as LY13  # noqa: E402,F401
import s07_comet_night_layout14 as LY14  # noqa: E402
import s07_comet_night_flank11 as FL  # noqa: E402
import s07_comet_night_forest11 as F11  # noqa: E402
V7.conifer_bands7 = F11.conifer_bands7     # individual clumped conifers instead of stacked bands

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
        self.truck = 0.13 * W                        # lateral truck amplitude (town plane)
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
        tw_T, tw_M = RL.twilight8(Wp, Hp, y_h, H, xs, ys, 0.36 * Wp, seed=8)
        sky = sky * (1 - tw_M[..., None]) + tw_T * tw_M[..., None]
        del tw_T, tw_M
        tg = np.exp(-((xs - 0.27 * Wp) / (0.17 * Wp)) ** 2) * np.exp(-np.clip(y_h - ys, 0, None) / (0.05 * H)) *             (ys < y_h + 2)
        sky += tg[..., None] * np.array([0.22, 0.11, 0.04], np.float32)
        # round 18: the warm afterglow rises softly into the base of the Milky Way (painterly warm/cool sky)
        mwg = np.exp(-((xs - 0.84 * Wp) / (0.22 * Wp)) ** 2) * np.exp(-np.clip(y_h - ys, 0, None) / (0.2 * H)) *             (ys < y_h + 2)
        sky += mwg[..., None] * np.array([0.1, 0.035, 0.03], np.float32)
        wg2 = np.exp(-np.clip(y_h - ys, 0, None) / (0.32 * H)) * (ys < y_h + 2)
        sky += wg2[..., None] * np.array([0.035, 0.012, 0.0], np.float32)
        del mwg, wg2
        # cool cyan skyglow around / below the comet
        cg = np.exp(-((xs - hx) / (0.28 * Wp)) ** 2) * np.exp(-((ys - (hy + 0.05 * H)) / (0.3 * H)) ** 2)
        sky += cg[..., None] * np.array([0.0, 0.05, 0.09], np.float32)

        # Milky Way rising from the right-hand horizon, clear of the comet
        mw, dust, dens = MW12.milky_way12(Wp, Hp, (Wp / 2 + 0.46 * W, y_h + 0.03 * H), (Wp / 2 + 0.07 * W, my - 0.15 * H),
                                        width=0.085 * H, seed=11, strength=0.72, bend=0.03, s=s, core_u=0.3)
        ext = 1 - 0.6 * C.smoothstep(y_h - 0.35 * H, y_h, np.arange(Hp, dtype=np.float32))   # extinction
        mw *= ext[:, None, None]
        dens *= ext[:, None] ** 2
        # the Milky Way lives on its own celestial plate (it parallaxes against the land + twilight glow)
        self.mw = (mw - 0.07 * dust[..., None] * sky).astype(np.float32)

        # star hierarchy: fine dust concentrated in the band, many small coloured point stars (yn_05),
        # sparkle spikes reserved for a few bright ones
        st, phase = P.stars(Wp, Hp, 21, dens, n_base=1300, scale=0.45)
        st += V.fine_stars(Wp, Hp, 23, dens ** 1.3, int(115000 * Wp * Hp / (1920 * 1080)))
        st += R.big_glints(Wp, Hp, 29, 6)
        st += V8.coloured_stars(Wp, Hp, int(420 * Wp * Hp / (1920 * 1080)), s, seed=33)
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
        tb, tw0, tw1, twx = 0.38, 0.003, 0.17, 1.25
        # yn_05 split comet: hard white core line, striated cyan dust fan with brushed feathering, thin
        # broken magenta fringe + separated magenta/gold secondary band (s07_comet_night_comet8)
        # round 19: a true split tail - narrow straight COOL ion tail + wider curved WARM dust tail
        # (magenta -> pink -> gold) diverging from the nucleus with dark sky between them (s07_comet_night_comet19)
        self.frag = (hx + d1[0] * 0.035 * L, hy + d1[1] * 0.035 * L)
        d2 = _rot(d1, -8.5)
        t2 = CM19.ion_tail19(Wp, Hp, self.head, d2, 1.0 * L, W, s, curve=-0.03, seed=7, w_end=0.03)
        t1, u1, vn1, b1 = CM19.dust_tail19(Wp, Hp, self.head, d1, 0.98 * L, W, s, bend=tb, extra=0.32, w0=0.002,
                                           w1=0.078, off0=0.012, seed=19, n_dust=int(1400 * s + 200))
        # round 20: warm/cool split - the dust tail's outer half turns peach/gold near the nucleus (warm) against
        # the cyan ion tail (cool); the inner edge stays rose
        u1c = np.clip(u1, 0, 1.2)
        wk = (C.smoothstep(-0.4, 0.9, vn1) * (1 - C.smoothstep(0.3, 0.85, u1c)) * (vn1 < 8)).astype(np.float32)
        wk2 = (C.smoothstep(-1.2, 0.2, vn1) * (1 - C.smoothstep(0.05, 0.5, u1c)) * (vn1 < 8)).astype(np.float32)
        lum1 = t1.max(-1, keepdims=True)
        t1 = t1 * (1 - 0.8 * wk[..., None]) + lum1 * np.array([1.0, 0.62, 0.3], np.float32) * (0.85 * wk[..., None])
        t1 = t1 * (1 - 0.35 * wk2[..., None]) + lum1 * np.array([1.0, 0.56, 0.5], np.float32) * (0.35 * wk2[..., None])
        del lum1, wk, wk2, u1c
        tail = t1 + t2
        # separate tight glows in each tail's own colour (no wide cyan veil filling the gap)
        halo = F.fast_blur(t2, 0.012 * W) * 0.35 + F.fast_blur(t1, 0.012 * W) * 0.25
        tail = tail + halo
        tail += CM19.head_fringe(Wp, Hp, self.head, W, s)
        del halo, t2
        head_img = CM.comet_head8(Wp, Hp, self.head, s, W)
        head_img += CM.comet_head8(Wp, Hp, self.frag, s, W, amt=0.12)
        rng = np.random.default_rng(404)
        sp = []
        for k in range(0):
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
        cglow = np.exp(-dcen / 0.1)[..., None] * np.array([0.02, 0.045, 0.09], np.float32)
        fl_ = F.anime_flare(Wp, Hp, hx, hy, intensity=0.22, tint=(0.6, 0.8, 1.0), rays=4,
                            ray_len=0.025, ghosts=0.0, halo=0.15, streak=0.15, glow=0.25, rot=0.4)
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
        self.sky_par = 0.3          # comet plate lags the land by 30% of the tilt
        self.star_par = 0.42        # stars + Milky Way lag more (farthest plane)
        self.cloud_par = 0.1        # clouds lag least (nearest sky plane)

        # ------------------------------------------------------------------ night clouds (yn_05)
        # painted cauliflower clusters, torn streamers and scraps; one big bank low on the right, low
        # stratus scraps sitting in the warm twilight (warm undersides). Frame fractions at t=0.
        def fp(fx_, fy_):
            return Wp / 2 + (fx_ - 0.5) * W, my + fy_ * H
        # yn_05 painted night clouds: clustered broken banks, torn diagonal streamers along the comet's
        # direction, crisp moonlit tops, flat body, lost translucent undersides (s07_comet_night_cloud13)
        # round 16: geometric moonlit cloudlets (lit top plane, darker body, lobe-on-lobe, translucent
        # lost undersides, tail / Milky Way local tints) - s07_comet_night_cloud16
        # round 17: FLAT painted shapes (s07_comet_night_cloud17 + layout17): noise-torn silhouettes built from
        # a few big lobes, hard-edged flat value planes (pale moonlit top, flat blue-grey body, translucent
        # blue-violet base), cauliflower bumps on the sky side only, torn thick streaks (no bead chains),
        # crisp violet tops + flat orange undersides on the afterglow clouds
        # round 18: soft painted clouds from smooth metaball + billow fields at full resolution (no lattice noise,
        # no quantised planes): crisp AA scalloped moonlit tops, feathered lost undersides, soft value masses;
        # twilight banks with violet-grey bodies and a hot rim-lit underside (s07_comet_night_cloud18)
        # round 19: soft painted clouds - fewer, bigger, cumulus -> cirrus; moonlit plane only on the top edge,
        # flat body, lost 20-40 px undersides; twilight banks rim-lit only on the horizon side, hazed
        # (s07_comet_night_cloud19 + layout19)
        # round 20: painted yn_05 cloud CLUSTERS (lobed clumps, star gaps, satellites, broad moonlit plane, painted
        # shadow dabs, dry-brush undersides) + yn_02 flat-based twilight rows with a wide graded hot rim
        # (s07_comet_night_cloud20).  Two plates: near night clouds / far twilight rows, each with its own
        # parallax, wind drift and truck response.
        # round 21: the night banks are repainted as clustered cauliflower masses (s07_comet_night_cloud21): exact
        # AA silhouettes (no stepped pixels / matte fringe), a crisp moonlit crest plane, bodies falling off into
        # navy with lost translucent bases, a thin warm rim toward the comet.  The scattered cloudlets and the
        # slab on the tail are gone.  The far twilight rows stay (cloud20).  Plates are PREMULTIPLIED.
        tw_layout = [c for c in K20.layout20(Wp, W, H, my) if c['kind'] == 'tw']
        plates_ = K20.clouds20(Wp, Hp, tw_layout, H, (hx, hy), sky, full_tail, seed=20, sun_x=0.3 * Wp)
        tw_pl = K21.crisp_twilight(plates_[1], s)          # crisp lit tops (they read smeary)
        tw_pl[..., :3] *= tw_pl[..., 3:4]
        nt_pl = K21.night_clouds21(Wp, Hp, K21.layout21(Wp, W, H, my), H, (hx, hy), tail_glow=full_tail)
        # (plate, parallax lag, wind drift px/s as a fraction of W, truck factor)
        self.cloud_plates = []
        for pl_, par_, dr_, tk_ in ((tw_pl, 0.13, 0.004, 0.03), (nt_pl, 0.05, 0.011, 0.1)):
            rr_ = np.nonzero(pl_[..., 3].max(1) > 0.002)[0]
            if len(rr_):
                self.cloud_plates.append((pl_, (int(rr_.min()), int(rr_.max()) + 1), par_, dr_, tk_))
        del plates_

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
        for ip_, (pl, fpar) in enumerate(TR.range_plates(Wp, H, W, y_h, b0, Hl, cometL, cache_dir=cache,
                                                         light_dir=-0.85)):
            if ip_ == 2:
                # painted couloirs / rock masses / strata on the near flanks (not the valley centre)
                reg_ = lambda x: np.clip((0.5 * Wp - x) / (0.06 * Wp), 0, 1) +                     np.clip((x - 0.72 * Wp) / (0.06 * Wp), 0, 1)  # noqa: E731
                pl = FL.paint_flanks(pl, hx, Wp, H, s, seed=3, region=reg_)
            # crisp comet-light rim + snow glints on the comet-facing ridges, violet shadow sides
            pl = RL16.ridge_light16(pl, (hx, hy - b0), Wp, H, s, seed=10 + ip_, rim_amt=(0.6, 0.85, 1.0)[ip_],
                                 shadow_amt=(0.3, 0.45, 0.6)[ip_], glints=int((20, 35, 45)[ip_]))
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
                               _hex('#35599a'), mist_amt=0.28, back_haze=0.18, rim_col=(0.42, 0.66, 0.95),
                               forest_bias=0.3)
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
        layers.append((mid, 0.6))
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
                w = rng.uniform(0.008, 0.02) * W * scl
                h = rng.uniform(0.0045, 0.0105) * H * scl
                two = rng.random() < 0.3
                if two:
                    h *= 1.55
                oh = w * rng.uniform(0.05, 0.14)
                by_ = by
                by = by_ + rng.uniform(-0.0025, 0.0) * H * scl       # hand-placed: not one ruled row
                top = by - h
                dark_house = rng.random() < 0.28                      # some houses unlit
                walls_.append([(x, by), (x, top), (x + w, top), (x + w, by)])
                kind = rng.random()
                if kind < 0.5:            # gable end facing us
                    rh = rng.uniform(0.22, 0.62) * w
                    ap = (x + w * 0.5, top - rh)
                    roofs_.append([(x - oh, top + 0.5), (ap[0], ap[1]), (x + w + oh, top + 0.5)])
                    lit_.append([ap, (x + w + oh, top + 0.5), (ap[0], top + 0.5)])
                    edges_.append([(x - oh, top), ap, (x + w + oh, top)])
                elif kind < 0.85:         # hip / side-gable roof (ridge along the street)
                    rh = rng.uniform(0.14, 0.36) * w
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
                        if rng.random() < (0.0 if dark_house else 0.5):
                            wide = rng.random() < 0.3
                            ww = max((0.0036 if wide else 0.0022) * W * scl, 1.0)
                            wh = max(0.0028 * H * scl, 1.0)
                            wx0 = x + (k + 0.5) * w / nwin - ww / 2
                            wy0 = by - (h / floors) * (fl_i + 0.72)
                            col = np.array(win_pal[rng.choice(len(win_pal), p=win_p)])
                            win_rects.append((wx0, wy0, wx0 + ww, wy0 + wh, col, rng.uniform(0.6, 1.8)))
                by = by_
                x += w + 2 * oh + rng.uniform(0.0, 0.011) * W
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
        # round 20: warm window light spilling onto the ground / shore in front of the lit houses
        for (x0, y0, x1, y1, col, it) in win_rects[::2]:
            C.splat(limg, (x0 + x1) / 2, y_h - 0.0025 * H, 7 * s, np.asarray(col, np.float32) * 0.8 + 0.2 *
                    np.array([1.0, 0.6, 0.3], np.float32), 0.05 * it)
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
        # round 13: repaint the plates - flat painted rock/snow planes (no faceted bevels), aerial perspective
        # on the back ranges, forests as near-flat value masses with only a treeline edge of tree tips
        hz_ = sky[int(y_h - 0.25 * H), int(0.2 * Wp):int(0.8 * Wp)].mean(0)
        xf_ = np.arange(Wp, dtype=np.float32) / Wp
        xm_ = 1 - ((xf_ > 0.09) & (xf_ < 0.47)).astype(np.float32)
        xm_ = cv2.GaussianBlur(xm_[None, :], (0, 0), 10 * s + 1)[0]
        new_ = []
        for k_, (l_, f_) in enumerate(self.layers):
            # round 18: stronger aerial haze on the distant ranges + crisp afterglow rim on the crests
            # round 19: facets melted into broad painted value masses, warm afterglow wash on the sun-facing
            # slopes, violet on the turned-away ones; far ranges hazier, lower contrast, softer silhouettes
            if k_ == 0:
                l_ = FT.flatten_range(l_, s, haze=hz_, haze_amt=0.5, contrast=0.42)
                l_ = MT19.paint_masses(l_, s, H, Wp, 0.3 * Wp, soft=0.75, sigma=5.0, glow=0.18, violet=0.1,
                                       edge_soft=1.6, haze=hz_, haze_amt=0.15)
                l_ = RG18.warm_rim(l_, 0.3 * Wp, Wp, H, s, amt=0.3, width=2.2, seed=1, violet=0.0)
            elif k_ == 1:
                l_ = FT.flatten_range(l_, s, haze=hz_, haze_amt=0.3, contrast=0.65)
                l_ = MT19.paint_masses(l_, s, H, Wp, 0.3 * Wp, soft=0.6, sigma=4.0, glow=0.25, violet=0.18,
                                       edge_soft=0.9, haze=hz_, haze_amt=0.08)
                l_ = RG18.warm_rim(l_, 0.3 * Wp, Wp, H, s, amt=0.6, width=2.0, seed=2, violet=0.1)
            elif k_ == 2:
                l_ = FT.flatten_range(l_, s)
                l_ = MT19.paint_masses(l_, s, H, Wp, 0.3 * Wp, soft=0.45, sigma=3.5, glow=0.38, violet=0.3)
                l_ = RG18.warm_rim(l_, 0.3 * Wp, Wp, H, s, amt=1.0, width=2.4, seed=3, violet=0.2)
            elif k_ == 4:
                l_ = FT.flatten_forest(l_, H, s)
                # round 20: treeline detail only where the crest catches sky light, lost dark edges below
                l_ = RP20.paint_ridge(l_, H, s, 0.3 * Wp, seed=41, keep=0.7, dark=0.18, masses=0.06,
                                      rim=(0.05, 0.05, 0.06))
            elif k_ == len(self.layers) - 1:
                l_ = FT.flatten_forest(l_, H, s, xmask=xm_)
                l_ = RP20.paint_ridge(l_, H, s, 0.3 * Wp, seed=43, keep=0.6, dark=0.3, masses=0.08,
                                      rim=(0.03, 0.04, 0.07))
            new_.append((np.ascontiguousarray(l_).astype(np.float32), f_))
        # round 21: hard AA ridgelines (no mushy halo), painted couloirs / snow ribs on the ranges (the far range
        # was flat lilac triangles), crisper value edges, and one lifted mist band across the range bases
        hx_ = self.head[0]
        for k_, (hard_, amt_, dep_, per_, cr_) in enumerate(((2.6, 0.8, 0.06, 14, 0.5), (3.0, 0.9, 0.07, 16, 0.6),
                                                              (3.5, 0.7, 0.09, 21, 0.8))):
            l_, f_ = new_[k_]
            l_ = RG21.couloirs(RG21.harden(l_, hard_), s, H, Wp, 0.3 * Wp, hx_, seed=1 + k_, amt=amt_, depth=dep_,
                               period=per_)
            l_ = RG21.crisp(l_, cr_, 2.0 + 0.25 * k_, s)
            new_[k_] = (np.ascontiguousarray(l_).astype(np.float32), f_)
        fa_ = new_[4][0][..., 3]
        ftop_ = np.argmax(fa_ > 0.5, axis=0).astype(np.float32)
        ftop_[fa_.max(0) < 0.5] = fa_.shape[0]
        new_.insert(3, (RG21.lifted_mist(Wp, fa_.shape[0], ftop_, H, s, 0.3 * Wp, hx_, seed=5), 0.36))
        self.cloud_layer += 1
        self.layers = new_
        # half-res copies for the per-frame reflection band
        self.layers_h = [(cv2.resize(l, (Wp // 2, (Hl - b0) // 2), interpolation=cv2.INTER_AREA), f)
                         for (l, f) in self.layers]
        self.glow_h = cv2.resize(self.town_glow, (Wp // 2, (Hl - b0) // 2), interpolation=cv2.INTER_AREA)

        # ------------------------------------------------------------------ reflection plate
        # round 18: celestial plates are mirrored where they sit late in the tilt (they lag the land by sky_par);
        # the clouds are mirrored per frame in _lake (own drift + parallax, masked by the mirrored land)
        dyc = self.sky_par * self.tilt * 0.8
        self.refl_dyc = dyc
        Msh = np.array([[1, 0, 0], [0, 1, dyc]], np.float32)

        def shv(im):
            return cv2.warpAffine(im, Msh, (Wp, Hp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # the mirrored tail is broken up by the water: softened, dimmer (no hard CG band near the camera)
        cel = shv(self.mw + self.comet_static + cv2.GaussianBlur(full_tail, (0, 0), 12.0 * s + 0.5) * 0.4)
        full = self.sky + cel
        refl = cv2.GaussianBlur(full, (0, 0), sigmaX=0.8 * s + 0.2, sigmaY=1.8 * s + 0.3)
        soft = C.blur(cel, 7 * s + 0.5) * 0.45
        del cel
        refl += soft
        stw = shv(self.stars)
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
        self.refl_land_a = np.zeros((Hp, Wp), np.float32)
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
        hyr = hy + dyc
        self.lkc = V4.glitter_column(Hlk, Wp, hx, 0.0, 2 * y_h - hyr - self.lk0, H, s,
                                     (0.5, 0.78, 1.0), (0.05, 0.12, 0.24))
        self.lkc += streaks([(hx, hyr, np.array([0.6, 0.85, 1.0]), 1.6, 8 * s)])
        self.lkc += V4.tail_glitter(Hlk, Wp, (hx, hyr), d1, L, y_h, self.lk0, H, s, (0.35, 0.62, 1.0), amt=0.5,
                                    bend=0.2)
        # the tail's glitter columns are separate narrow strips: merge them into one soft wash
        self.lkc = cv2.GaussianBlur(self.lkc, (0, 0), sigmaX=5.0 * s + 0.5, sigmaY=0.5)
        del xk, yk, spill

        # ------------------------------------------------------------------ water textures
        self.T1 = P.tile_noise(256, 41, 4.0, 20.0)
        self.T2 = P.tile_noise(256, 43, 7.0, 30.0)
        self.T3 = P.tile_noise(256, 45, 10.0, 10.0)
        self.T4 = P.tile_noise(256, 47, 20.0, 60.0)
        # round 13: hand-placed ripple strokes (broken horizontal dashes, dark wave backs + pale sky-lit crests)
        rs_ = np.random.default_rng(1313)
        NS = 1024
        ts = np.zeros((NS, NS), np.float32)
        for i_ in range(1500):
            y_ = rs_.uniform(0, NS)
            x_ = rs_.uniform(0, NS)
            ln_ = rs_.uniform(10, 80) * (1 + 1.5 * rs_.random() ** 4)
            v_ = -1.0 if rs_.random() < 0.65 else 1.0
            hh_ = rs_.uniform(0.6, 1.4)
            for dx_ in (-NS, 0, NS):
                # long, thin, pointed dash (a brush flick), tapered at both ends
                cv2.ellipse(ts, (int((x_ + dx_) * 4), int(y_ * 4)), (int(ln_ * 2), int(hh_ * 4)), 0, 0, 360,
                            v_ * rs_.uniform(0.6, 1.0), -1, cv2.LINE_AA, shift=2)
        # tapered ends: soften horizontally only
        self.TS = cv2.GaussianBlur(ts, (0, 0), sigmaX=2.0, sigmaY=0.6)
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

        # round 21: near lakeside cedar - a foreground wipe-by rising from the bottom-right early in the tilt
        self.fg_spr, self.fg_anchor = FG21.cedar_sprite(W, H, seed=7)
        self.meteors = [(0.55, 0.45, 0.2, my + 0.14 * H, 152, 0.14, 1.0),
                        (2.35, 0.35, 0.86, my + 0.42 * H, 140, 0.11, 0.7),
                        (3.9, 0.5, 0.3, my + 0.6 * H, 158, 0.13, 1.0)]
        self.xs_f, self.ys_f = C.grid(W, H)

    # ---------------------------------------------------------------------------------------------
    def cam(self, t):
        """Round 19: asymmetric eased move - gentle ease-in, the bulk of the tilt in the first half, then a long
        slow settle (velocity ~ u (1-u)^2, i.e. a regularised incomplete beta(2, 3)): ~69% of the travel by
        mid-shot, ~31% in the second half."""
        u = min(max(t / DURATION, 0.0), 1.0)
        e23 = 6 * u ** 2 - 8 * u ** 3 + 3 * u ** 4        # I_u(2, 3)
        e5 = u * u * u * (u * (6 * u - 15) + 10)            # smootherstep: firmer ease-in and a longer settle
        e = 0.5 * e23 + 0.5 * e5
        z = 1.0 + 0.05 * e                                  # gentle push-in, same ease
        cy = self.my + self.H / 2 + self.tilt * e
        # round 20: lateral truck on the same ease + a slow constant crab so the planes separate from frame 1
        tr = self.truck * (e - 0.5) + 0.025 * self.W * (u - 0.5)
        return cy, z, e, tr

    def _refl_lag(self, t):
        """Reflection trails the tilt by a few px while the camera moves (0 at rest)."""
        dt_ = 0.35
        e1 = self.cam(t)[2]
        e0 = self.cam(max(t - dt_, 0.0))[2]
        return 0.9 * (e1 - e0) / dt_ * DURATION / 1.4 * 0.007 * self.H

    def _vlag(self, f, e):
        """Crane component: far land planes sink slightly behind the nearer ones as the camera lowers."""
        return 0.1 * max(0.0, 0.6 - f) * self.tilt * e

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
        # celestial plates (stars, Milky Way, comet) move less than the land during the tilt/crane
        Mc = M.copy()
        Mc[1, 2] -= self.sky_par * self.tilt * e
        # round 19 multi-plane: stars + Milky Way (farthest) lag more than the comet, the clouds lag least
        Ms = M.copy()
        Ms[1, 2] -= self.star_par * self.tilt * e
        yh_f = (self.y_h - M[1, 2]) * z
        r0 = int(min(max(math.floor(yh_f), 0), H))       # first lake row
        rs = min(r0 + 2, H)                                # sky/land rows to render
        u = t / DURATION

        img = np.zeros((H, W, 3), np.float32)
        met = np.zeros((H, W, 3), np.float32)
        clA = None
        tail_full = None
        if rs > 0:
            sk = self._warp_rows(self.sky, M, rs, cv2.BORDER_REPLICATE)
            stv = self._warp_rows(self.stars, Ms, rs)
            M2 = Ms.copy()
            M2[:, :2] *= 2
            hr = (rs + 1) // 2
            ph = cv2.warpAffine(self.phase, M2, (W // 2, hr), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT)
            tw = 0.8 + 0.2 * np.sin(t * (1.6 + 2.4 * ph) + ph * 40.0)
            tw = cv2.resize(tw, (W, hr * 2), interpolation=cv2.INTER_NEAREST)[:rs]
            sk += stv * tw[..., None]
            sk += self._warp_rows(self.mw, Ms, rs)
            sk += self._warp_rows(self.comet_static, Mc, rs)
            # tail: slowly lengthening, with gently flowing streamers
            tx0, ty0 = self.tbox
            th_, tw_ = self.tail.shape[:2]
            fx0 = int(max((tx0 - Mc[0, 2]) * z - 2, 0)); fx1 = int(min((tx0 + tw_ - Mc[0, 2]) * z + 2, W))
            fy0 = int(max((ty0 - Mc[1, 2]) * z - 2, 0)); fy1 = int(min((ty0 + th_ - Mc[1, 2]) * z + 2, rs))
            if fx1 > fx0 and fy1 > fy0:
                Mt = np.array([[1 / z, 0, Mc[0, 2] - tx0 + fx0 / z], [0, 1 / z, Mc[1, 2] - ty0 + fy0 / z]], np.float32)
                tl = cv2.warpAffine(self.tail, Mt, (fx1 - fx0, fy1 - fy0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT)
                aux = cv2.warpAffine(self.tail_aux, Mt, (fx1 - fx0, fy1 - fy0),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
                uu, vn, bd = aux[..., 0], aux[..., 1], aux[..., 2]
                Lt = 0.64 + 0.4 * (0.25 * u + 0.75 * C.ease_in_out_sine(u))
                grow = 1 - C.smoothstep(Lt - 0.28, Lt, uu)
                # streamers flow outward along the tail (knots travel away from the nucleus)
                flow = (np.sin(vn * 9.0 + uu * 22.0 - t * 2.2) * 0.6 + np.sin(vn * 19.0 - uu * 13.0 - t * 1.5) * 0.4)
                tl = tl * grow[..., None] + (bd * flow * grow)[..., None] * np.array([0.7, 0.4, 0.55], np.float32) * 0.18
                sk[fy0:fy1, fx0:fx1] += tl
                tail_full = np.zeros((rs, W, 3), np.float32)
                tail_full[fy0:fy1, fx0:fx1] = tl
            # fragment sparkle (slow, smooth glint)
            nsp = len(self.sparks)
            for i, (x, y, it, col) in enumerate(self.sparks):
                fxs, fys = (x - Mc[0, 2]) * z, (y - Mc[1, 2]) * z
                if 0 <= fys < rs:
                    k = 0.72 + 0.28 * math.sin(t * (1.3 + 0.37 * i) + i * 2.1)
                    C.splat(sk, fxs, fys, 0.9 * s + 0.35, np.asarray(col, np.float32), 2.0 * it * k)
                    C.splat(sk, fxs, fys, 4.0 * s + 0.5, np.asarray(col, np.float32), 0.12 * it * k)
                    if i >= nsp - self.n_front:
                        C.splat(sk, fxs, fys, 1.3 * s + 0.4, np.asarray(col, np.float32), 2.2 * it * k)
                        V4.cross_flare(sk, fxs, fys, (12 + 7 * it) * s, 0.5 * s + 0.3, col, 0.8 * it * k, rot=0.3)
            # hot nucleus: short cross-flare
            hfx0, hfy0 = (self.head[0] - Mc[0, 2]) * z, (self.head[1] - Mc[1, 2]) * z
            if -40 < hfy0 < rs + 40:
                tw_k = 0.92 + 0.08 * math.sin(t * 2.1)
                CM9.star_flare(sk, hfx0, hfy0, s, amt=tw_k, rot=0.3)
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
                self._streak(met, (hx - Mc[0, 2]) * z, (hy - Mc[1, 2]) * z, -dx, -dy, tail * z, 1.1 * s + 0.4,
                             br * max(env, fade_after * 0.25))
            sk += met[:rs]
            # painted clouds (in front of the comet): far twilight rows, then near night clouds, each plate with
            # its own parallax lag, wind drift and truck response (they read as separate planes)
            for (cpl, crow, cpar, cdr, ctk) in self.cloud_plates:
                Mcl = M.copy()
                Mcl[1, 2] -= cpar * self.tilt * e
                Mcl[0, 2] -= ctk * tr + cdr * W * t
                c0_ = int(max((crow[0] - Mcl[1, 2]) * z - 2, 0))
                c1_ = int(min((crow[1] - Mcl[1, 2]) * z + 2, rs))
                if c1_ > c0_:
                    Mcl[1, 2] += c0_ / z
                    cl = cv2.warpAffine(cpl, Mcl, (W, c1_ - c0_), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_CONSTANT)
                    a_ = cl[..., 3:4]
                    sk[c0_:c1_] = sk[c0_:c1_] * (1 - a_) + cl[..., :3]
                    if tail_full is not None:
                        # the tail streams across in front of the thin night clouds (yn_05): its light
                        # stays readable over them instead of the cloud reading as a sticker on top
                        sk[c0_:c1_] += tail_full[c0_:c1_] * a_ * 0.55
                    # painted clouds stay out of the bloom source (no halo rim)
                    if clA is None:
                        clA = np.zeros((H, W), np.float32)
                    clA[c0_:c1_] = np.maximum(clA[c0_:c1_], a_[..., 0])
            img[:rs] = sk
            # land planes with parallax
            fr0 = int(max(math.floor((self.b0 - M[1, 2]) * z), 0))
            if rs > fr0:
                band = img[fr0:rs]
                for k_, (lay, f) in enumerate(self.layers):
                    dft = self.cloud_drift * t if k_ == self.cloud_layer else 0.0
                    lw = self._warp_band(lay, M, f * tr + dft, self.b0 + self._vlag(f, e), fr0, rs)
                    band = F.over_rgba(band, lw)
                band += self._warp_band(self.town_glow, M, tr, self.b0, fr0, rs)
                img[fr0:rs] = band

        blink = max(0.0, math.sin(t * 2 * math.pi * 0.6)) ** 3
        fm = 0.6 * tr
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
            self._e = e
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
        hfx = (self.head[0] - Mc[0, 2]) * z
        hfy = (self.head[1] - Mc[1, 2]) * z
        if -0.2 * H < hfy < 1.1 * H:
            vx, vy = W / 2 - hfx, H / 2 - hfy
            for kk, rad, col, it in ((0.55, 0.012, (0.4, 0.7, 1.0), 0.04), (1.25, 0.03, (0.5, 0.9, 0.8), 0.025),
                                     (1.7, 0.018, (0.8, 0.6, 1.0), 0.03)):
                C.splat(img, hfx + vx * kk, hfy + vy * kk, rad * W, np.array(col, np.float32), it)

        # near cedar wipe-by: climbs ~1.4x faster than the town and slides out right with the truck
        tr_end = self.cam(DURATION)[3]
        sway_ = 0.0
        FG21.composite(img, self.fg_spr, 1.17 * W + 3.0 * (tr - tr_end) + sway_,
                       0.34 * H + (1 - e) * self.tilt * 1.4 * z, self.fg_anchor)
        img = self._reeds(img, t, e, z, 3.4 * tr - 0.12 * W)

        img = self._bloom(img, mask=clA)
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
            Mh = np.array([[1, 0, (f * tr + dft) / 2], [0, 1, self._vlag(f, self._e) / 2]], np.float32)
            lw = cv2.warpAffine(lay, Mh, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            comp = F.over_rgba(comp, lw)
            alpha = np.maximum(alpha, lw[..., 3])
        Mh = np.array([[1, 0, tr / 2], [0, 1, 0]], np.float32)
        comp += cv2.warpAffine(self.glow_h, Mh, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        comp = cv2.GaussianBlur(comp, (0, 0), sigmaX=0.6 * s + 0.1, sigmaY=1.4 * s + 0.15)
        # mirrored land a touch darker and softer than the originals
        comp = comp * (1 - 0.18 * alpha[..., None])
        Hb = self.Hl - self.b0
        big = cv2.resize(comp, (self.Wp, Hb), interpolation=cv2.INTER_LINEAR)
        ab = cv2.resize(alpha, (self.Wp, Hb), interpolation=cv2.INTER_LINEAR)
        big += self.rb_soft + self.rb_glint * (1 - ab[..., None])
        self.refl[self.b0:self.Hl] = big * self.refl_tint
        self.refl_land_a[self.b0:self.Hl] = ab

    def _bloom(self, img, threshold=0.75, knee=0.3, strength=0.35, mask=None):
        H, W = self.H, self.W
        q = 4
        small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
        lum = small.max(-1, keepdims=True)
        k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1) ** 2
        if mask is not None:
            ms = cv2.resize(mask, (W // q, H // q), interpolation=cv2.INTER_AREA)
            k = k * (1 - 0.92 * np.clip(ms, 0, 1))[..., None]
        br = small * k
        acc = np.zeros_like(br)
        for r, wt in ((0.004, 1.0), (0.012, 0.8), (0.035, 0.6), (0.09, 0.45)):
            acc += F.fast_blur(br, max(r * W / q, 0.6)) * wt
        acc /= 2.85
        # round 18: halation (a faint warm-tinted skirt around the hottest points) + a faint anamorphic glint
        hot = np.clip((lum - 1.3) / 1.2, 0, 1) ** 2
        if mask is not None:
            hot = hot * (1 - np.clip(ms, 0, 1))[..., None]
        hb = (small * hot).max(-1)
        ana = cv2.GaussianBlur(hb, (0, 0), sigmaX=0.05 * W / q, sigmaY=0.35) * 0.9 +             cv2.GaussianBlur(hb, (0, 0), sigmaX=0.15 * W / q, sigmaY=0.5) * 0.5
        acc += ana[..., None] * np.array([0.35, 0.6, 1.0], np.float32)
        hal = F.fast_blur(small * hot, max(0.006 * W / q, 0.6))
        acc += hal * np.array([1.0, 0.75, 0.7], np.float32) * 0.5
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
        N = 256.0
        Hr_, Wr_ = px.shape
        # smooth ripple fields are evaluated at half resolution and upsampled (they are band-limited by the
        # anti-alias factors anyway); the reflection / light look-ups stay full resolution
        hq = 2
        pxh = cv2.resize(px, ((Wr_ + 1) // hq, (Hr_ + 1) // hq), interpolation=cv2.INTER_LINEAR)
        pyh = cv2.resize(py, ((Wr_ + 1) // hq, (Hr_ + 1) // hq), interpolation=cv2.INTER_LINEAR)

        def rm(T, u, v):
            return cv2.remap(T, np.mod(u, N).astype(np.float32), np.mod(v, N).astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dyw = np.maximum(pyh - y_h, 0.25)
        D = (0.05 * H) / dyw
        X = (pxh - tr - self.Wp / 2) / dyw
        tu1 = X * 24.0 + t * 1.5
        tv1 = D * 700.0 - t * 5.0
        tu2 = X * 45.0 - t * 2.0
        tv2 = D * 1100.0 - t * 7.0
        n1 = rm(self.T1, tu1, tv1)
        n2 = rm(self.T2, tu2, tv2)
        dDdy = (0.05 * H) / (dyw * dyw) / z
        aa = np.clip(1.5 / np.maximum(dDdy * 700.0, 1e-3), 0, 1)
        nn = (n1 * 0.65 + n2 * 0.35) * aa
        A = 0.02 * dyw + 0.4 * s
        nh1 = rm(self.T1, X * 7.0 + t * 0.25, D * 900.0 - t * 4.0)
        nh2 = rm(self.T1, X * 22.0 + 40 - t * 0.4, D * 2100.0 - t * 6.0 + 17)
        nh1 = nh1 * np.clip(1.5 / np.maximum(dDdy * 900.0, 1e-3), 0, 1)
        nh2 = nh2 * np.clip(1.5 / np.maximum(dDdy * 2100.0, 1e-3), 0, 1)
        # sparse ripple patches: most of the lake is a calm mirror, wind-ripple breaks come in drifting
        # irregular patches (no regular scanline banding)
        gz = rm(self.T3, X * 4.0 + 11 + t * 0.15, D * 160.0 - t * 1.2 + 23)
        gate = C.smoothstep(0.2, 1.3, gz) * np.clip(1.5 / np.maximum(dDdy * 160.0, 1e-3), 0, 1)
        gate = 0.15 + 0.85 * gate
        nh1 = nh1 * gate
        nh2 = nh2 * gate
        dsx = nh2 * A * 0.22
        dsy = (nn * 0.3 * gate + nh1 * 1.0) * A
        fres = (0.66 + 0.26 * np.exp(-dyw / (0.14 * H))) * (1 + 0.06 * nh1 + 0.05 * nh2)
        dly = nh1 * A * 0.3
        dlx = nh2 * A * 0.6
        nb = rm(self.T2, tu2 * 1.7 + 30, tv2 * 2.3 + 70)
        # light columns break into tapering dashes (ripple crests), not continuous bars
        dsh = nb * aa + 0.6 * n2 * aa + 0.5 * nh1
        band = np.clip(0.12 + 1.1 * C.smoothstep(-0.3, 0.7, dsh), 0.05, 1.4)
        crest = nh2 + 0.35 * nh1
        glit = 0.12 + 0.5 * C.smoothstep(0.2, 0.9, crest) + 2.2 * C.smoothstep(0.95, 1.15, crest)
        sheen = np.exp(-dyw / (0.012 * H))
        # round 19: broad, slowly drifting swell shading - perspective-correct (fine + compressed at the far
        # shore, broad near the camera), anti-aliased so it fades out instead of aliasing into scanlines
        sw1 = rm(self.T3, X * 30.0 + 5 + t * 0.9, D * 320.0 - t * 2.6 + 9)
        sw2 = rm(self.T3, X * 70.0 + 70 - t * 1.3, D * 750.0 - t * 4.1 + 131)
        swl = sw1 * np.clip(2.5 / np.maximum(dDdy * 320.0, 1e-3), 0, 1) +             0.5 * sw2 * np.clip(2.5 / np.maximum(dDdy * 750.0, 1e-3), 0, 1)
        stack = np.dstack([dsx, dsy, fres, dly, dlx, band, glit, sheen, nn * A, gate, swl]).astype(np.float32)
        up = cv2.resize(stack, (Wr_, Hr_), interpolation=cv2.INTER_LINEAR)
        dsx, dsy, fres, dly, dlx, band, glit, sheen, nnA, gateF, swl = [up[..., k] for k in range(11)]

        sx = px + dsx
        sy = np.clip(2 * y_h - py + dsy + self._refl_lag(t), 0, self.Hp - 1)
        refl = cv2.remap(self.refl, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        # mirrored clouds: this frame's drift + parallax, hidden behind the mirrored land
        sxf, syf = sx.astype(np.float32), sy.astype(np.float32)
        lm = None
        for (cpl, crow, cpar, cdr, ctk) in self.cloud_plates:
            cox = ctk * tr + cdr * W * t
            coy = cpar * self.tilt * self._e
            clr = cv2.remap(cpl, sxf - np.float32(cox), syf - np.float32(coy), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            if clr[..., 3].max() > 0.002:
                if lm is None:
                    lm = cv2.remap(self.refl_land_a, sxf, syf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                ca = (clr[..., 3] * (1 - np.clip(lm, 0, 1)) * 0.92)[..., None]
                cfac = (1 - np.clip(lm, 0, 1)) * 0.92
                refl = refl * (1 - ca) + clr[..., :3] * cfac[..., None] * self.refl_tint * 0.9
        # round 19: reflections soften (and later darken) progressively away from the far shore - a vertical
        # water smear that grows toward the camera, instead of a sharp mirror cut into scanlines
        dyr = py[:, :1] - y_h
        kb = (C.smoothstep(0.015 * H, 0.35 * H, dyr) * 0.7)[..., None]
        rbl = cv2.GaussianBlur(refl, (0, 0), sigmaX=1.2 * s + 0.3, sigmaY=4.0 * s + 0.5)
        refl = refl * (1 - kb) + rbl * kb
        rows = np.clip((py[:, 0] - self.lk0).astype(np.int32), 0, self.water_grad.shape[0] - 1)
        wc = self.water_grad[rows][:, None, :]
        fr0 = 0.66 + 0.26 * np.exp(-np.maximum(py[:, :1] - y_h, 0.25) / (0.14 * H))
        out = wc * (1 - fr0[..., None]) + refl * fres[..., None]
        # broken light columns (town lights truck with the town; the comet streak stays with the sky)
        lky = (py - self.lk0 + dly).astype(np.float32)
        lkx = px + dlx
        lk = cv2.remap(self.lk, (lkx - tr).astype(np.float32), lky, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out += lk * (band * 1.35)[..., None]
        # comet glitter: ripple crests catch the light -> broken horizontal dashes along the column
        lc = cv2.remap(self.lkc, lkx.astype(np.float32), lky, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        out += lc * glit[..., None]
        # glints: tiny ripple facets catching the sky / comet light, twinkling and drifting slowly
        dywf = np.maximum(py - y_h, 0.25)
        Df = (0.05 * H) / dywf
        Xf = (px - tr - self.Wp / 2) / dywf
        gn = rm(self.T4, Xf * 30.0 + t * 0.6, Df * 900.0 - t * 3.5)
        gi = np.nonzero(gn > 2.3)
        if len(gi[0]):
            g_ = gn[gi]
            dD = (0.05 * H) / (dywf[gi] ** 2) / z
            aag = np.clip(1.2 / np.maximum(dD * 900.0 * 3, 1e-3), 0, 1)
            spk = C.smoothstep(2.3, 3.1, g_) * aag
            twk = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(t * 3.3 + 7.0 * g_ + 0.05 * gi[1]))
            rv = refl[gi]
            add = (spk * twk)[:, None] * (rv * 1.6 + rv.max(-1, keepdims=True) * np.array([0.3, 0.45, 0.7], np.float32))
            out[gi] += add
        out += sheen[..., None] * np.array([0.08, 0.14, 0.22], np.float32)
        # ---- round 13: broken ripple strokes + near-camera darkening (not a blurred mirror)
        ns_ = float(self.TS.shape[0])
        dyf = np.maximum(py - y_h, 0.25)
        Xs = (px - tr - self.Wp / 2) / dyf
        Ds = (0.05 * H) / dyf
        su = np.mod(Xs * 300.0 + t * 1.2, ns_).astype(np.float32)
        sv = np.mod(Ds * 1500.0 - t * 10.0, ns_).astype(np.float32)
        stk = cv2.remap(self.TS, su, sv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dDs = (0.05 * H) / (dyf * dyf) / z * 1500.0
        aas = np.clip(2.0 / np.maximum(dDs, 1e-3), 0, 1) ** 0.7
        # round 19: the stroke texture only lives in drifting wind patches and is much gentler (it read as
        # even scanlines across the whole lake)
        stk = stk * aas * (0.25 + 0.75 * np.clip((gateF - 0.15) / 0.85, 0, 1))
        dark_ = np.clip(-stk, 0, 1)
        lite_ = np.clip(stk, 0, 1)
        out = out * (1 - 0.16 * dark_[..., None]) + lite_[..., None] * (0.1 * out + np.array([0.02, 0.04, 0.07], np.float32))
        sws = np.clip(swl, -2.0, 2.0) * C.smoothstep(0.0, 0.08 * H, py - y_h)
        out = out * (1 + 0.1 * sws)[..., None] + np.clip(sws - 0.6, 0, None)[..., None] *             np.array([0.015, 0.03, 0.06], np.float32)
        # water darkens toward the camera (less sky reflected at steep view angles)
        near_ = C.smoothstep(0.06 * H, 0.5 * H, py - y_h)
        out = out * (1 - 0.45 * near_[..., None])
        # a thin dark shoreline shadow under the far bank
        out = out * (1 - 0.35 * np.exp(-np.maximum(py - y_h, 0) / (0.004 * H)))[..., None]
        if met.any():
            rr = np.arange(r0, H, dtype=np.float32)
            src_rows = np.clip(2 * yh_f - rr, 0, H - 1).astype(np.float32)
            mx = np.repeat(np.arange(W, dtype=np.float32)[None, :], len(rr), 0) + dsx * 2.0
            my_ = np.repeat(src_rows[:, None], W, 1) + nnA
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
        base_y = H * 1.01 + (1 - e) * self.tilt * 1.9 * z
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
            # gentle sway + a slow gust travelling across the reed bed (right to left)
            sway = 0.05 * math.sin(t * fr + ph) + 0.02 * math.sin(t * fr * 2.3 + ph * 1.7) +                 0.04 * math.sin(t * 1.1 + bx * 6.0) * (0.6 + 0.4 * math.sin(t * 0.5 + 1.0))
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
