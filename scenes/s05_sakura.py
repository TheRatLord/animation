"""s05_sakura - spring morning on a riverside embankment path lined with cherry trees in full bloom.

Composition: the path and its railing recede to a vanishing point at the right third (a small bridge sits
there); the blossom tunnel of the land-side tree row frames the right and the top; across the river a second
row of cherries glows backlit in front of a hazy city; the sky with a painted cumulus fills the upper left,
the morning sun peeks through a hanging branch in the top-left corner. Petals drift at several depths.
Camera: slow lateral dolly (true multi-plane parallax: every element is placed in 3D and shifted by 1/Z).
Round 2: canopies repainted as a few big painted blossom masses per tree (s05_sakura_mass: lit pink-white
tops, cool lavender undersides, crisp cast shadows between masses, warm transmitted light, lacy 5-petal edges
and sky pinholes); clouds from lib.clouds2; Japanese lettering on the signboards (s05_sakura_signs); denser
facade detail; static paper texture (no per-frame grain); camera travel ~40% slower.
Round 3: all cherries are hand-painted 2D tree cards (s05_sakura_tree2d): forked trunks with warm bark, rim
light, cool bounce and vertical fissures whose limbs vanish into 3-6 big blossom masses per tree (lit
pink-white tops, crisp lavender undersides, warm transmitted hems, lacy 5-petal florets, sky pinholes,
blossom clumps where limbs enter); varied far-bank row with gaps; camera-facing signboards with real
Japanese; painted cumulus with lavender shadows + silver lining; clearer vertical-streak water reflections
with parallax-correct sampling, sparse glints, hanaikada ribbons; varied sharp sunflecks.
Round 4: blossom masses repainted as layered flower clumps (s05_sakura_bloom): 3-5 separated masses per tree,
lit crescent caps following each clump's lacy edge, value-band edges broken by 5-petal florets, clustered floret
texture, irregular jagged pinholes without outlines; limbs truncated inside solid blossom, re-tapered, clusters
on the wood, and any wood cut off from the trunk by a long hidden stretch repainted as blossom (no floating
sticks); pink-white palette; signboards: no pseudo-glyph blocks, fewer/larger boards, 2-4 character words, bold
glyphs, too-small boards left plain; path shade cool blue-violet with value variation + warm bounce and a
coin-broken edge; far town with warmer materials, local-contrast/saturation push; painted railing pipes
(s05_sakura_rail) with cool shadow side, warm rim, sliding specular glints; stronger canal ripple, warmer glitter.
Round 5: canopies from s05_sakura_canopy2 - 3-5 big overlapping value masses per crown (lit pink-white crown, mid
pink body, lavender underside with warm bounce), floret detail only on silhouettes / mass boundaries (interior ~30%),
crisp cast shadows, 8-15 real sky holes per near tree, peach transmitted glow + 2-3 px warm rims on sun-facing
edges, a lavender skirt + steeper limbs so no bare forks show; far bank with varied materials, warmer shade, eave
shadows, stronger aerial fade; bigger camera-facing signs, noren, more vending machines; land-side houses in canopy
shade; far paving pulled down ~10%, tighter bloom.
Round 7: canopies painted as clusters of small blossom clumps (s05_sakura_clumps.Painter4: flat cel bands, lit
pink-white caps, lavender-pink undersides, crisp contact shadows, sky showing through the gaps, AA florets on the
silhouette); dark 'front bark' limb streaks removed; near petals fewer, smaller and crisp (tiny CoC); far-bank
town gets painted accent lines, rain stains, wall mottle and extra clutter (s05_sakura_facade); silver lining on
the sky cumulus; more water / rail glints; ACCENT: an eased tilt-up (t 1.45-4.25 s, 0.2 H) that follows a gust of
petals lifted into the sky (canvas with headroom above the frame, sub-pixel crop).
Round 8: every cherry crown repainted by s05_sakura_crown5.Painter5 - hundreds of small AA flower clusters
(scalloped floret heads) grouped into 0.3-0.55 m clumps that carry the value (warm-white lit tops, mid pink,
lavender undersides, per-cluster contact shadows), open sky gaps in the upper crown with pink-gold backlit
edges, a cluster-built shadow back layer in the lower crown, wood only as coherent trunk-connected limbs (no
floating stubs), blossom clumps where limbs enter the crown, cherry bark with lenticels / fissures / lichen,
warm sun rim + cool shade; sun-facing silver lining + stronger value structure on the sky cumulus; facade
curtains / sashes / laundry / sill stains + denser rooftop clutter; small signs get simple legible words;
coherent glitter band on the canal under the sun (s05_sakura_glint) + warm specular line on the railing.
Round 10 (final-panel notes): BOTH cherry rows repainted by s05_sakura_r10 - 5-8 big blossom clumps per tree on an
arching crown with real sky gaps, each clump 3-6 overlapping bunch-edged lobes lit from the top-left sun (hot/lit
pink-white tops, mid pink body, cool mauve underside, crisp overlap + cast shadows), 5-petal floret detail only on
the silhouettes; trunks/limbs are tapered, gently curving dark plum-brown silhouettes with a thin warm sun-side rim
(no kinks, no CG bark). Tilt-up now a symmetric sine ease over the whole shot (0.26 H, peak speed ~45% lower).
Round 11 (reviewer FAIL notes): both cherry rows repainted by s05_sakura_r11 - trunks with random lean / C, S or
straight curves that fork into 2-3 scaffold limbs + sub-forks feeding 5-8 flattened, drooping, tiered clumps (no
balls); each clump painted as flat planes from the sun direction (lit pink-white top-left, crisp terminator broken
into blossom steps, mauve underside), varied-size florets + twig sprays only on the silhouette (sparse on the shadow
side), limbs crossing open sky gaps (no punched holes), warm 2-3 px trunk rim + bark value break; far row varied in
size / shape / lean with stronger aerial haze; land-side houses: clean painted canopy-shadow line (no camo mottle);
tilt-up extended to 0.62 H with a symmetric linear+sine ease (peak ~1.3x mean) ending on open sky with cirrus and
petals; round flare ghosts only.
Round 12 (reviewer FAIL notes): both cherry rows repainted by s05_sakura_r12 - recursive, continuously tapering,
gently curving limbs (Leonardo-rule forks, no kinks or stubs) whose ends and fan of fine twigs disappear INTO the
clump undersides; clumps are irregular unions of tilted / drooping rounded lobes (no flat bottoms) shaded as one
form with soft AA lit / mid / mauve planes that wrap under each lobe; floret-built silhouettes (5-petal blossoms,
bunches, sprays) with lacy overlap edges; clumps spaced for real sky gaps + loose blossom twigs across them;
wood: dark tapered silhouettes, sun-side-only 1-3 px gold rim, knots / darker crotches; premultiplied card fix
(no dark fringes); far row varied per tree.
Round 13 (5cm/s study): both cherry rows rebuilt by s05_sakura_r13 - stout flared trunks splitting low into wide
arching scaffolds with alternating laterals; blossom born on the wood as hundreds of small clusters of AA floret
dabs grouped into masses (lit per MASS: pink-white tops, warm rose mids, lavender / violet undersides, a shadow
body behind covered clusters, warm transmitted sun-side fringes), fine wood tucked inside the blossom so only
limbs cross it; far row denser with slimmer trunks; sky replaced by one continuous gradient over the tilt-up
headroom (cerulean zenith -> pale warm horizon, tight sun glow) instead of the flat cobalt.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import core as C, sky as S, fx as F  # noqa: E402
from lib import clouds2 as K2  # noqa: E402
import s05_sakura_raster as R  # noqa: E402
import s05_sakura_env as E  # noqa: E402
import s05_sakura_trees as T  # noqa: E402
import s05_sakura_branch as BR  # noqa: E402
import s05_sakura_fx as SFX  # noqa: E402
import s05_sakura_grass as G  # noqa: E402
import s05_sakura_canopy as CN  # noqa: E402
import s05_sakura_mass as MS  # noqa: E402
import s05_sakura_signs as SG  # noqa: E402
import s05_sakura_tree2d as T2  # noqa: E402
import s05_sakura_bloom as BL  # noqa: E402
import s05_sakura_rail as RL  # noqa: E402
import s05_sakura_canopy2 as C2  # noqa: E402
import s05_sakura_canopy3 as C3  # noqa: E402
import s05_sakura_dress as DR  # noqa: E402
import s05_sakura_motion as MO  # noqa: E402
import s05_sakura_clumps as CL  # noqa: E402
import s05_sakura_facade as FC  # noqa: E402
import s05_sakura_crown5 as P5  # noqa: E402
import s05_sakura_glint as GL  # noqa: E402
import s05_sakura_r9 as R9  # noqa: E402
import s05_sakura_r9env as R9E  # noqa: E402
import s05_sakura_r10 as R10  # noqa: E402
import s05_sakura_r11 as R11  # noqa: E402
import s05_sakura_r12 as R12  # noqa: E402
import s05_sakura_r13 as R13  # noqa: E402

DURATION = 5.0
TRAVEL = 2.0          # metres of lateral camera travel over the shot (round 2: ~40% slower)
HERO_Z = 5.5          # effective depth of the hand-painted foreground branch
CRANE = 0.62          # tilt-up (fraction of H) following the petals into the sky (round 10: longer, gentler)
CRANE_T = (0.0, 5.0)      # round 10: symmetric sine ease over the whole shot (peak speed ~45% lower)

SPRING_SKY = dict(stops=[(0.0, '#2459c4'), (0.22, '#3272d4'), (0.45, '#5595e2'), (0.66, '#8dbdee'),
                         (0.84, '#bfdcf3'), (1.0, '#e2eff6')],
                  sun_glow='#fff4e4', sun_glow_amt=0.16, below='#e2eff6', band=('#f0f7fa', 0.5))

# painted cumulus: noon palette with lavender shadow masses; cauliflower levels of varied scale
# round 8: lit face stepped down from the hot peak (hi) so the HDR silver lining reads against it, cooler
# lavender-blue shadow masses for a clear lit / mid / shadow value structure
CU_PAL = dict(hi=(0.97, 0.94, 0.9), lit=(0.9, 0.87, 0.84), lit_lo='#cdc4d2', mid='#bcb3cf', shade='#9d9cd6',
              deep='#7f83c6', refl='#b0bbe8', bounce='#d8c8d2', rim=(1.75, 1.6, 1.25), haze='#c3e1f6', edge_dark=0.05)
CU_LEV = ((0.14, 0.4, 0.95, 1.2, 2.4, 0.12, 0.45), (0.05, 0.13, 0.85, 1.0, 2.2, 0.3, 0.6),
          (0.02, 0.045, 0.6, 1.2, 3.0), (0.012, 0.02, 0.2, 1.6, 4.0))

# blossom palettes (deep shade, shade, mid, lit, hot)
PAL_NEAR = np.array([[0.4, 0.22, 0.54], [0.88, 0.47, 0.66], [1.0, 0.72, 0.82], [1.03, 0.88, 0.9],
                     [1.1, 1.02, 0.96]], np.float32)
PAL_FAR = np.array([[0.66, 0.42, 0.66], [0.97, 0.63, 0.77], [1.02, 0.76, 0.84], [1.04, 0.89, 0.91],
                    [1.08, 1.0, 0.96]], np.float32)
PAL_HERO = np.array([[0.6, 0.34, 0.56], [0.86, 0.54, 0.72], [0.99, 0.78, 0.84], [1.03, 0.93, 0.94],
                     [1.1, 1.04, 1.02]], np.float32)
# repaint palettes (deep, shade, mid, lit, hot) - cool lavender shade, clean pink mids, pink-white light
MASS_NEAR = np.array([[0.5, 0.34, 0.62], [0.8, 0.5, 0.75], [0.96, 0.65, 0.81], [1.0, 0.82, 0.88],
                      [1.03, 0.93, 0.94]], np.float32)
MASS_FAR = np.array([[0.58, 0.42, 0.7], [0.84, 0.56, 0.78], [0.97, 0.68, 0.83], [1.0, 0.83, 0.89],
                     [1.03, 0.93, 0.94]], np.float32)
MASS_KW_NEAR = dict(per_mass=4, kmin=3, kmax=9, seed=1)
MASS_KW_FAR = dict(per_mass=5, kmin=1, kmax=3, seed=2)
GLOW_NEAR = (1.0, 0.66, 0.72)
GLOW_FAR = (1.0, 0.74, 0.78)
# round 4 blossom palette: pink-white lit tops (never grey-white), clean pink mids, cool lavender shade,
# warm peach transmitted light
BLOSSOM_PAL = dict(lit=np.array([1.0, 0.75, 0.83], np.float32), hot=np.array([1.03, 0.83, 0.87], np.float32),
                   mid=np.array([0.96, 0.66, 0.8], np.float32), lav=np.array([0.7, 0.58, 0.82], np.float32),
                   deep=np.array([0.56, 0.45, 0.72], np.float32), trans=np.array([1.0, 0.78, 0.74], np.float32),
                   rim=np.array([1.16, 1.0, 0.84], np.float32),
                   bounce=np.array([0.9, 0.64, 0.66], np.float32), glow=np.array([1.06, 0.8, 0.64], np.float32))
BARK = np.array([[0.05, 0.036, 0.06], [0.3, 0.19, 0.15]], np.float32)
BARK_FAR = np.array([[0.2, 0.18, 0.3], [0.44, 0.34, 0.38]], np.float32)     # distant trunks: lighter, bluer


def _ease(u):
    return 0.65 * u + 0.35 * C.ease_in_out_sine(u)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        cam = self.cam = E.Cam(W, H)
        f = cam.f
        # ---------------------------------------------------------------- sun
        self.sun_xy = (0.17 * W, 0.1 * H)
        self.M = int(math.ceil(CRANE * H)) + 2          # canvas headroom above the frame for the tilt-up
        sd = cam.screen_to_dir(*self.sun_xy)          # camera coords (x right, y up, z fwd)
        self.sun_dir = sd
        Lu = sd[0] * cam.ca - sd[2] * cam.sa
        Ls = sd[0] * cam.sa + sd[2] * cam.ca
        self.L_path = (Lu, sd[1], Ls)
        # ---------------------------------------------------------------- sky
        self._build_sky()
        # ---------------------------------------------------------------- environment
        self._build_env()
        # ---------------------------------------------------------------- trees
        self._build_trees()
        self.grass = G.build(cam, W, H, self.emx, self.ground)
        # grass tufts growing in front of the near trunk feet (the flare sinks into the grass)
        rg = np.random.default_rng(404)
        self.grass_feet = {}
        for k_, (u, s) in enumerate(self.near_bases):
            Zt = cam.to_cam(u, 0.0, s)[2]
            if Zt > 60:
                continue
            tf = []
            for k in range(26):
                a = rg.uniform(-2.6, 0.5)            # mostly on the camera side of the trunk
                rad = rg.uniform(0.3, 0.7)
                tf.append((u + rad * math.cos(a), s + rad * math.sin(a), rg.uniform(0.7, 1.1)))
            self.grass_feet[round(float(Zt), 4)] = G.build(cam, W, H, self.emx, self.ground, seed=77 + k_,
                                                            tufts=tf, nb=2)
        # ---------------------------------------------------------------- reflection source for the water
        self._build_reflection()
        # ---------------------------------------------------------------- petals
        self._init_petals()
        # ---------------------------------------------------------------- optics / sparkles
        self.flare = SFX.SunFlare(W, H)
        # round 11: round, soft ghosts only (no hexagonal aperture ghosts)
        self.flare.ghosts = [(g[0], g[1], g[2], 0, g[4] * 0.8, g[5]) for g in self.flare.ghosts]
        self.paper = self._paper()
        self._sky_rows = int(0.64 * H)
        yy = np.arange(H, dtype=np.float32)[:, None, None]
        kk = np.clip((yy - self.cam.hy) / (H - self.cam.hy), 0, 1)
        wall = np.array([0.66, 0.66, 0.78], np.float32) * (1 - 0.25 * kk) + np.array([0.0, 0.02, 0.06]) * kk
        xx = np.arange(W, dtype=np.float32)[None, :, None]
        # stone courses sloping toward the vanishing point
        ph = (yy - self.cam.hy + 0.25 * (xx - 0.6 * W)) / (0.035 * H + 0.06 * (yy - self.cam.hy))
        line = np.clip(1 - np.abs(ph - np.round(ph)) / 0.06, 0, 1) * 0.12
        self._wall = np.broadcast_to(wall * (1 - line), (H, W, 3)).astype(np.float32).copy()
        self._hero_rows = min(H, int(0.62 * H))
        a = self.far_a
        self._far_rgba = np.zeros((H, a.shape[1], 4), np.float32)
        self._far_rgba[..., 3] = a
        self._far_ia = (1.0 / np.maximum(a, 1e-4))[..., None].astype(np.float32)
        Wc = self.spot.shape[1]
        self._spot_q = cv2.resize(self.spot, (Wc // 4, H // 4), interpolation=cv2.INTER_AREA)
        self.dapple = MO.Dapple(self.spot, self.cam.hy, H, W, self.emx)
        rng = np.random.default_rng(77)
        self._rail_s = np.sort(rng.uniform(1.2, 45.0, 34))
        self._rail_keep = np.arange(34) % 3 == 0
        self._rail_sz = rng.uniform(0.5, 1.2, 34)
        self._rail_in = rng.uniform(0.5, 1.3, 34)
        # water sparkles: dense in the sun's glitter path, a scatter of sky glints elsewhere
        gx = (self.w_x - (self.sun_xy[0] + self.emx)) / (0.05 * W)
        gy = (self.w_y - (2 * self.cam.hy - self.sun_xy[1])) / (0.25 * H)
        gw = np.exp(-gx * gx - gy * gy) * self.w_a * np.clip((self.w_Z - 7.0) / 10.0, 0.08, 1.0)
        p = gw / gw.sum()
        n1 = 34
        i1 = rng.choice(len(p), n1, replace=False, p=p)
        p2 = np.clip((self.w_Z - 9.0) / 10.0, 0.0, 1.0) + 1e-6
        i2 = rng.choice(len(p), 22, replace=False, p=p2 / p2.sum())
        idx = np.concatenate([i1, i2])
        self._ws_px = self.w_x[idx]
        self._ws_py = self.w_y[idx]
        self._ws_x = self.w_x[idx].astype(np.float64)
        self._ws_y = self.w_y[idx].astype(np.float64)
        zz = self.w_Z[idx]
        self._ws_sz = W * np.clip(0.012 * 8.0 / zz, 0.002, 0.012) * rng.uniform(0.6, 1.3, len(idx))
        self._ws_in = np.concatenate([rng.uniform(0.9, 1.6, n1), rng.uniform(0.35, 0.7, len(i2))])
        # round 8: coherent glitter band under the sun (ripple dashes + warm sheen) replaces the random crosses
        self.glit = GL.Glitter(W, H, self.w_x, self.w_y, self.w_Z, self.w_a, self.emx, self.sun_xy[0], self.cam.hy)
        self.glit.inten = self.glit.inten * 1.7            # round 9: brighter glitter band under the sun
        sh = self.glit.sheen(self.water_a, None)
        self._sheen = np.ascontiguousarray(sh[:, self.emx:self.emx + W]).astype(np.float32)
        sc_ = int(np.clip(self.sun_xy[0] + self.emx, 0, self.far_invz.shape[1] - 1))
        self._sheen_invz = self.far_invz[:, sc_].astype(np.float32)

    # ================================================================== sky
    def _build_sky(self):
        W, H = self.W, self.H
        self.smx = int(0.04 * W)
        pw, ph = W + 2 * self.smx, H
        sun_p = (self.sun_xy[0] + self.smx, self.sun_xy[1])
        hz = self.cam.hy / H
        sky = S.sky_gradient(pw, ph, SPRING_SKY, horizon=hz, sun=sun_p, sun_radius=0.4)
        # painted cumulus (lib.clouds2): a hero cumulus in the open sky between the branch and the canopy, two
        # smaller hazier ones, a low bank sinking behind the city, faint cirrus high up
        oy_ = 0.0
        # round 3: painted with the clouds2 Painter directly - varied cauliflower scale along the edge (a few
        # big heads, mid florets, a sprinkle of small ones), lavender shadow masses, strong silver lining
        u_ = W / 1920.0
        pal = K2.palette(CU_PAL)
        P = K2.Painter(pw, ph, sun_p, pal, u_, seed=31, sun_z=0.15)
        for (cx_, by_, cw_, ch_, sd_, dist_) in ((0.31 * pw, 0.45 * H, 0.27 * W, 0.25 * H, 12, 0.0),
                                                (0.47 * pw, 0.43 * H, 0.1 * W, 0.07 * H, 14, 0.3)):
            R9E.cumulus9(K2, P, cx_, by_, cw_, ch_, seed=sd_, pal=K2.mix_palette(CU_PAL, 'noon_far', dist_), n=5,
                         haze=0.3 * dist_, levels=CU_LEV, firm=1.0, soften=0.08, term_noise=0.1, brush=0.02)
        P.lining(1.8, rim_px=3.0, halo=0.6)
        hero = self._silver(P.rgba(), 2.6)
        P = K2.Painter(pw, ph, sun_p, pal, u_, seed=33, sun_z=0.35)
        for (cx_, by_, cw_, ch_, sd_, dist_) in ((0.1 * pw, 0.42 * H, 0.18 * W, 0.06 * H, 21, 0.55),
                                                (0.84 * pw, 0.4 * H, 0.12 * W, 0.045 * H, 23, 0.7)):
            K2.cumulus(P, cx_, by_, cw_, ch_, seed=sd_, pal=K2.mix_palette(CU_PAL, 'noon_far', dist_), n=3,
                       haze=0.35 * dist_, levels=CU_LEV, sun_jitter=8.0)
        P.lining(1.0, rim_px=2.5)
        far = self._silver(P.rgba(), 0.6)
        bank = K2.horizon_bank_plate(pw, ph, 0.47 * H, 0.05 * H, sun=sun_p, preset='noon', seed=41, rows=3,
                                     haze=0.35, haze_color='#d8eaf6', shrink=0.4, layer_haze=0.1)
        cir = K2.cirrus_plate(pw, ph, preset='noon', seed=4, region=(0.02, 0.26), angle=-10, density=0.35,
                              opacity=0.35)
        # headroom above the frame (tilt-up): the sky deepens toward the zenith, cloud plates padded clear
        M = self.M
        # round 13: one continuous painted gradient over headroom + frame (no flat cobalt): cerulean at the
        # zenith, clear mid blue, pale cyan, near-white warm haze at the horizon, a warm glow around the sun
        sky = self._sky13(pw, ph + M, M + self.cam.hy, (sun_p[0], sun_p[1] + M))

        def padp(p_):
            p_ = p_.copy()
            fade = np.clip(np.arange(p_.shape[0], dtype=np.float32) / (0.05 * H), 0, 1)
            p_[..., 3] *= fade[:, None]
            z = np.zeros((M,) + p_.shape[1:], np.float32)
            z[..., :3] = p_[:1, :, :3]
            return np.concatenate([z, p_], 0)
        # round 11: high cirrus / mackerel wisps in the headroom - the tilt-up ends on open sky with texture
        hi = K2.cirrus_plate(pw, ph + M, preset='noon', seed=12, region=(0.02, 0.5 * M / (ph + M) + 0.02),
                             angle=-8, density=0.35, opacity=0.3)
        self.clouds = S.CloudDrift([(hi, 0.0015, 0.02), (padp(cir), 0.0012, 0.02), (padp(bank), 0.0006, 0.02),
                                    (padp(far), 0.0012, 0.03, 0.0005), (padp(hero), 0.002, 0.04, 0.0008)])
        self.sky = sky

    def _sky13(self, w, h, hz_row, sun):
        W = self.W
        r = np.arange(h, dtype=np.float32)
        e = np.clip((hz_row - r) / hz_row, -0.2, 1.0)
        stops = [(-0.2, (0.93, 0.95, 0.97)), (0.0, (0.94, 0.96, 0.96)), (0.06, (0.84, 0.92, 0.97)),
                 (0.16, (0.64, 0.82, 0.96)), (0.32, (0.45, 0.7, 0.94)), (0.6, (0.32, 0.59, 0.9)),
                 (1.0, (0.24, 0.49, 0.85))]
        xs = [q[0] for q in stops]
        col = np.stack([np.interp(e, xs, [q[1][c] for q in stops]) for c in range(3)], -1).astype(np.float32)
        img = np.broadcast_to(col[:, None, :], (h, w, 3)).copy()
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d2 = ((xx - sun[0]) ** 2 + (yy - sun[1]) ** 2) / (W * W)
        g = (0.2 * np.exp(-d2 / 0.006) + 0.06 * np.exp(-d2 / 0.06))[..., None]
        img = img + (np.array([1.0, 0.97, 0.9], np.float32) - img) * np.clip(g, 0, 1)
        # a gentle lateral lift toward the sun side (the sky is brighter under the sun)
        lat = (1.0 - xx / w)[..., None] * 0.04
        img = img + (1.0 - img) * lat
        return img.astype(np.float32)

    def _silver(self, rgba, amt):
        """round 7: a crisp 2-3 px silver lining along the cloud's upper / outer silhouette against the sky,
        strongest where the paint just inside is in lavender shadow (thin translucent edges glow) and on the
        sun-facing side; lower (base) edges get none."""
        u = self.W / 1920.0
        A = np.ascontiguousarray(rgba[..., 3])
        r = max(1, int(round(2.2 * u)))
        er = cv2.erode(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ring = np.clip(A - er, 0, 1)
        # upward-facing edges only (sky above them)
        up = np.clip(A - np.roll(cv2.GaussianBlur(A, (0, 0), 3 * u), -int(6 * u), 0), 0, 1)
        up = cv2.GaussianBlur(up, (0, 0), 2 * u)
        up = np.clip(up * 3, 0, 1)
        inner = cv2.GaussianBlur(rgba[..., :3], (0, 0), 5 * u)
        lum = inner.mean(-1)
        shade = np.clip((0.95 - lum) / 0.25, 0, 1)
        # round 8: only the silhouette that faces the sun (upper-left) gets the lining
        Ab = cv2.GaussianBlur(A, (0, 0), 12 * u)
        gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        hh, ww = A.shape
        ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
        sx, sy = self.sun_xy[0] + self.smx, self.sun_xy[1]
        dx, dy = sx - xs, sy - ys
        dn = np.sqrt(dx * dx + dy * dy) + 1e-6
        facing = -(gx * dx + gy * dy) / (gn * dn)
        fw = np.clip((facing + 0.15) / 0.6, 0, 1)
        k = ring * np.maximum(up, 0.6 * fw) * fw * (0.55 + 0.45 * shade) * amt
        k = cv2.GaussianBlur(k, (0, 0), 0.6 * u)[..., None]
        out = rgba.copy()
        out[..., :3] = out[..., :3] * (1 - k) + np.array([1.22, 1.2, 1.16], np.float32) * k
        return out

    # ================================================================== environment
    def _build_env(self):
        W, H = self.W, self.H
        cam = self.cam
        ss = 2
        self.emx = int(0.3 * W)
        Wc = W + 2 * self.emx
        Hs, Ws = H * ss, Wc * ss
        B, poles = E.make_boxes()
        # keep the far skyline low (the sky is the hero): cap building tops at an elevation angle
        e_max = (cam.hy - 0.34 * H) / cam.f
        for i in range(len(B)):
            if B[i, 6] != 0:
                continue
            X, Y, Z = cam.to_cam(B[i, 1], 0.0, B[i, 4])
            cap = E.Y_FAR + 0.0 + e_max * Z * (0.75 + 0.25 * ((i * 7919) % 13) / 13.0)
            B[i, 3] = max(B[i, 2] + 3.0, min(B[i, 3], cap))
        B = E.rooftops(B)
        B = E.riverside_details(B)
        B = DR.dress(B)
        B = FC.dress2(B)
        B = E.box_bboxes(cam, B, self.emx, ss)
        mat = np.zeros((Hs, Ws), np.int8)
        Zo = np.zeros((Hs, Ws), np.float32)
        uo = np.zeros((Hs, Ws), np.float32)
        Yo = np.zeros((Hs, Ws), np.float32)
        so = np.zeros((Hs, Ws), np.float32)
        face = np.zeros((Hs, Ws), np.int8)
        bid = np.zeros((Hs, Ws), np.int32)
        E.raycast(Hs, Ws, float(ss), cam.f, cam.cx + self.emx, cam.hy, cam.ca, cam.sa, B, mat, Zo, uo, Yo, so,
                  face, bid)
        out = np.zeros((Hs, Ws, 4), np.float32)
        layer = np.zeros((Hs, Ws), np.int8)
        spot = np.zeros((Hs, Ws), np.float32)
        Lu, LY, Ls = self.L_path
        haze = np.array([0.8, 0.87, 0.96], np.float32)
        E.shade_env(mat, Zo, uo, Yo, so, face, bid, B, cam.f, float(ss), Lu, LY, Ls, haze, 260.0, 0.85, out, layer,
                    self.sun_xy[0], spot)
        SG.paint_signs(out, bid, face, uo, Yo, so, B, ss, min_px=150, wscale=W / 1920.0)
        # round 7: painted accent lines where surfaces meet + rain-stain weathering (no clean CG boxes)
        FC.weather(out, mat, uo, Yo, so, face, layer, Zo)
        FC.ink(out, mat, bid, face, Zo, layer, ss, W / 1920.0)
        # round 9: grime under eaves / sills, damp plinths, panel variation (both banks), ink on the land-side
        # houses, painted clumped hedge
        R9E.weather9(out, mat, bid, face, uo, Yo, so, layer, Zo, B)
        R9E.ink9(out, mat, bid, face, Zo, layer, ss, W / 1920.0)
        R9E.paint_hedge(out, mat, bid, face, uo, Yo, so, B, ss, Zo=Zo)
        # far-bank town: push local contrast + saturation (crisper facades, stronger warm/cool split)
        fm = ((layer == 0) & (mat == 5)).astype(np.float32)
        if fm.any():
            rgbf = out[..., :3]
            sg_ = 5.0 * ss * W / 1920.0
            base = cv2.GaussianBlur(rgbf * fm[..., None], (0, 0), sg_) /                 np.maximum(cv2.GaussianBlur(fm, (0, 0), sg_), 1e-3)[..., None]
            det = rgbf + (rgbf - base) * 0.6
            lum = det.mean(-1, keepdims=True)
            det = lum + (det - lum) * 1.25
            det = np.clip(det * 0.97, 0, 1.2)
            out[..., :3] = rgbf + (det - rgbf) * fm[..., None]
        # round 9: distant blocks sink into the aerial haze (after the local-contrast push)
        R9E.far_haze(out, mat, layer, Zo)

        # land-side houses behind the hedge (round 6): read as SOLID painted buildings, not a hazy ghost -
        # stronger local contrast, a touch more saturation, warmer / darker values against the pale sky
        hm = (layer == 5).astype(np.float32)
        if hm.any():
            rgbf = out[..., :3]
            sg_ = 4.0 * ss * W / 1920.0
            base = cv2.GaussianBlur(rgbf * hm[..., None], (0, 0), sg_) /                 np.maximum(cv2.GaussianBlur(hm, (0, 0), sg_), 1e-3)[..., None]
            det = rgbf + (rgbf - base) * 0.3
            lum = det.mean(-1, keepdims=True)
            det = lum + (det - lum) * 1.1
            det = det * np.array([0.9, 0.85, 0.8], np.float32)
            out[..., :3] = rgbf + (det - rgbf) * hm[..., None]

        def down(a):
            return cv2.resize(a, (Wc, H), interpolation=cv2.INTER_AREA)

        def plate_for(mask):
            m = mask.astype(np.float32)
            pm = out[..., :3] * m[..., None]
            a = down(m)
            rgb = down(pm)
            return np.dstack([rgb / np.maximum(a, 1e-4)[..., None], a]).astype(np.float32), a

        self.ground, _ = plate_for(layer == 1)
        # round 5: the sunlit paving toward the vanishing point was blowing out - pull the far lit values down
        #          ~10% (keeps the joints / shadow edges readable under the bloom)
        yy_ = (np.arange(H, dtype=np.float32)[:, None] - cam.hy) / (0.22 * H)
        g_ = self.ground[..., :3]
        lum_ = g_.mean(-1)
        kd = (0.13 * np.exp(-yy_ * yy_) * (yy_ > -0.05)) * np.clip((lum_ - 0.62) / 0.25, 0, 1)
        self.ground[..., :3] = g_ * (1 - kd[..., None])
        self.spot = down(spot * (layer == 1))
        self.overlay, _ = plate_for(layer == 2)
        # land-side hedge and houses: own cards, moved with the exact parallax of a vertical plane
        # parallel to the path (per-column inverse mapping) so they never tear away from the ground
        self.cards = []
        for lay, u_pl in ((5, 8.6), (4, 6.0)):
            rgba, ca_ = plate_for(layer == lay)
            rows = np.nonzero(ca_.max(1) > 0)[0]
            if len(rows) == 0:
                continue
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            pm = np.dstack([rgba[r0:r1, :, :3] * ca_[r0:r1, :, None], ca_[r0:r1]]).astype(np.float32)
            xs = np.arange(Wc, dtype=np.float64) - self.emx + 0.5
            ru = (xs - cam.cx) / cam.f * cam.ca - cam.sa
            invz = np.where(ru > 1e-4, ru / u_pl, 0.0)
            self.cards.append((pm, r0, invz))
        far_mask = (layer == 0) & (mat > 0)
        m = far_mask.astype(np.float32)
        self.far_pm = down(out[..., :3] * m[..., None])            # premultiplied, water excluded
        wmask = (mat == 2).astype(np.float32)
        self.water_a = down(wmask)
        self.far_a = np.clip(down(m) + self.water_a, 0, 1)
        # depth maps (plate res)
        Zf = np.where(mat > 0, Zo, 1e5).astype(np.float32)
        invz = down(np.where(far_mask | (mat == 2), 1.0 / Zf, 0).astype(np.float32))
        cov = np.maximum(down(m + wmask), 1e-4)
        invz = invz / cov
        # dilate inverse depth horizontally so silhouettes move with their object
        k = max(3, int(0.03 * W) | 1)
        invz = cv2.dilate(invz, np.ones((1, k), np.uint8))
        # never shift the far/water plate beyond its margin (the nearest water sits under the railing)
        invz = np.minimum(invz, 0.95 * self.emx / (cam.f * TRAVEL * 0.5))
        self.far_invz = invz.astype(np.float32)
        # water pixel lists (plate res)
        wu = down(np.where(mat == 2, uo, 0)) / np.maximum(self.water_a, 1e-4)
        ws = down(np.where(mat == 2, so, 0)) / np.maximum(self.water_a, 1e-4)
        wz = down(np.where(mat == 2, Zo, 0)) / np.maximum(self.water_a, 1e-4)
        yy, xx = np.nonzero(self.water_a > 0.003)
        self.w_y, self.w_x = yy.astype(np.int64), xx.astype(np.int64)
        self.w_u = wu[yy, xx].astype(np.float32)
        self.w_s = ws[yy, xx].astype(np.float32)
        self.w_Z = wz[yy, xx].astype(np.float32)
        self.w_a = self.water_a[yy, xx].astype(np.float32)
        ry = (cam.hy - (yy + 0.5)) / cam.f
        rx = (xx + 0.5 - self.emx - cam.cx) / cam.f
        self.w_ry = (np.abs(ry) / np.sqrt(rx * rx + ry * ry + 1)).astype(np.float32)
        # far-wall base line per column (reflection axis)
        cols = np.arange(Wc) + 0.5 - self.emx
        rxc = (cols - cam.cx) / cam.f
        ru = rxc * cam.ca - cam.sa
        Zw = np.where(ru < -1e-4, E.U_FARWALL / np.minimum(ru, -1e-4), 1e5)
        self.ywb = (cam.hy - cam.f * E.Y_WATER / Zw).astype(np.float32)
        self.ywb = np.minimum(self.ywb, cam.hy + (cam.hy - 0) * 0 + H)  # guard
        self.poles = poles
        self._wires()

    def _wires(self):
        """Power lines between the far-bank poles (drawn into the overlay plate)."""
        cam = self.cam
        W, H = self.W, self.H
        Wc = W + 2 * self.emx
        q = 4
        m = np.zeros((H * q, Wc * q), np.uint8)
        for dy_, du in ((9.62, -0.8), (9.62, 0.8), (8.62, -0.6), (8.62, 0.6)):
            for (u, s0), (_, s1) in zip(self.poles[:-1], self.poles[1:]):
                tt = np.linspace(0, 1, 40)
                s = s0 + (s1 - s0) * tt
                Y = E.Y_FAR + dy_ - 0.9 * 4 * tt * (1 - tt)
                X, Yc, Z = cam.to_cam(u + du + 0 * s, Y, s)
                if Z.min() < 1:
                    continue
                x, y = cam.proj(X, Yc, Z, 0, self.emx)
                wdt = max(1, int(round(0.025 * cam.f / Z.mean() * q)))
                pts = (np.stack([x, y], 1) * q).astype(np.int32)
                cv2.polylines(m, [pts], False, 255, wdt, cv2.LINE_AA)
        m = cv2.resize(m.astype(np.float32) / 255.0, (Wc, H), interpolation=cv2.INTER_AREA)
        col = np.array([0.42, 0.45, 0.58], np.float32)
        a0 = self.overlay[..., 3]
        rgb = self.overlay[..., :3] * a0[..., None] * (1 - m[..., None]) + col * m[..., None] * 0.9
        a = a0 + m * 0.9 * (1 - a0)
        self.overlay = np.dstack([rgb / np.maximum(a, 1e-4)[..., None], a]).astype(np.float32)

    # ================================================================== trees
    def _build_trees(self):
        """Round 3: every cherry is a hand-painted 2D tree card (s05_sakura_tree2d) placed at its trunk depth:
        forked trunks growing into 3-6 big painted blossom masses, lacy 5-petal edges, sky pinholes."""
        W, H = self.W, self.H
        cam = self.cam
        f = cam.f
        ss = 2
        rng = np.random.default_rng(55)
        sx, sy = self.sun_xy
        # ---- near bank row (land side of the path), crowns leaning over the path
        s_list = [11.5, 18.5, 25.5, 32.5, 40.0, 47.5, 55.5, 64.0, 73.0, 83.0, 94.0, 106.0, 119.0, 146.0, 161.0,
                  177.0, 195.0, 215.0, 237.0, 260.0]
        near_bases = []
        self.bins_near = []
        hcol_n = np.array([0.86, 0.87, 0.98], np.float32)
        for i, s in enumerate(s_list):
            u = 2.5 + rng.uniform(-0.2, 0.3)
            near_bases.append((u, s))
            X, Y, Z = cam.to_cam(u, E.Y_PATH, s)
            bx, by = cam.proj(X, Y, Z)
            ss = 2
            k = f * ss / Z
            tr_rng = np.random.default_rng(1000 + i)
            sc = rng.uniform(0.95, 1.08)
            big_ = Z < 20
            tree = R13.Tree13(tr_rng, scale=1.15 * sc, lean=-1.0, spread=1.25, big=Z < 45, hmul=1.1)
            ldx, ldy = sx - bx, (sy - by) * 1.0
            ln = math.hypot(ldx, ldy) + 1e-6
            wpx = W / 1920.0                      # 1080p pixel in output px
            # round 13: branch-borne blossom clusters (s05_sakura_r13)
            pnt = R13.Painter13(k, sun_dir=(ldx / ln, ldy / ln), dab_min=1.6 * ss * wpx, rim_px=1.6 * ss * wpx,
                                wood_rim_px=1.6 * ss * wpx, detail=1.0 if Z < 60 else 0.8)
            card, ox, oy = pnt.paint(tree, tr_rng)
            hz = 0.72 * (1.0 - math.exp(-Z / 200.0))
            T2.haze_card(card, hz, hcol_n)
            rgba, x0, y0 = T2.card_to_screen(card, ox, oy, float(bx), float(by), ss)
            self.bins_near.append(dict(rgba=rgba, ox=x0, oy=y0, z=float(Z), hero=False, foot=float(by - y0),
                                       phase=float(0.9 * i + 0.4 * math.sin(3.7 * i))))
        self.n_near = len(s_list)
        self.bins_near_sorted = sorted(self.bins_near, key=lambda b: -b['z'])
        self._contact_shadows(near_bases)
        self.near_bases = near_bases
        # ---- hand-painted hero branch hanging into the top-left corner (2D, nearest plane)
        self.hero, self.hero_ox, self.hero_oy = BR.render(W, H, self.sun_xy, top=0.1 + CRANE + 0.02)
        a = self.hero[..., 3:4]
        self.hero_pm = np.concatenate([self.hero[..., :3] * a, a], -1).astype(np.float32)
        # ---- far bank row: varied crowns, irregular spacing with gaps that show the town behind
        self.bins_far = []
        hcol_f = np.array([0.8, 0.85, 0.97], np.float32)
        s = 20.0
        i = 0
        while s < 420:
            u = -32.3 + rng.uniform(-0.6, 0.6)
            X, Y, Z = cam.to_cam(u, E.Y_FAR, s)
            bx, by = cam.proj(X, Y, Z)
            sc = 1.35 * rng.choice([0.7, 0.85, 1.0, 1.15, 1.3], p=[0.15, 0.25, 0.3, 0.2, 0.1]) * rng.uniform(0.92, 1.08)
            if bx > -0.3 * W and bx < 1.25 * W:
                ss = 2
                k = f * ss / Z
                tr_rng = np.random.default_rng(5000 + i)
                # round 11: varied size / shape per tree (height, spread, clump size, lean all random)
                tree = R13.Tree13(tr_rng, lean=float(tr_rng.choice([-1.0, 1.0], p=[0.35, 0.65])),
                                  scale=0.9 * sc, big=False, spread=tr_rng.uniform(0.8, 1.2),
                                  hmul=tr_rng.uniform(0.85, 1.2), density=1.5, trunk=0.72)
                ldx, ldy = sx - bx, sy - by
                ln = math.hypot(ldx, ldy) + 1e-6
                wpx = W / 1920.0
                pnt = R13.Painter13(k, sun_dir=(ldx / ln, ldy / ln), dab_min=1.3 * ss * wpx, dab_m=0.07,
                                    rim_px=1.3 * ss * wpx, wood_rim_px=1.0 * ss * wpx, detail=0.7, far=True)
                card, ox, oy = pnt.paint(tree, tr_rng)
                hz = 0.62 * (1.0 - math.exp(-Z / 120.0))       # round 11: stronger aerial fade with distance
                T2.haze_card(card, hz, hcol_f)
                rgba, x0, y0 = T2.card_to_screen(card, ox, oy, float(bx), float(by), ss)
                self.bins_far.append(dict(rgba=rgba, ox=x0, oy=y0, z=float(Z), hero=False))
            i += 1
            gap = rng.uniform(6.5, 9.5) * (0.6 + 0.4 * sc) * 1.35
            if rng.random() < 0.28:
                gap += rng.uniform(5.0, 11.0)            # a real gap: the buildings show through
            s += gap

    def _contact_shadows(self, bases):
        """soft dark pools on the ground around the trunk feet (ground plate, static)"""
        cam = self.cam
        W, H = self.W, self.H
        Wc = W + 2 * self.emx
        m = np.zeros((H, Wc), np.float32)
        for (u, s) in bases:
            for rad, a in ((0.9, 0.45), (0.45, 0.6)):
                th = np.linspace(0, 2 * np.pi, 48)
                uu = u + rad * np.cos(th) * 1.2
                sv = s + rad * np.sin(th) - 0.25
                X, Y, Z = cam.to_cam(uu, E.Y_PATH + 0 * uu, sv)
                if Z.min() < 0.5:
                    continue
                x, y = cam.proj(X, Y, Z, 0, self.emx)
                m = np.maximum(m, C.polygon_mask(Wc, H, np.stack([x, y], 1)) * a)
        m = C.blur(m, 0.004 * W)
        g = self.ground
        g[..., :3] *= (1 - m[..., None] * np.array([0.75, 0.72, 0.55], np.float32))
        self.ground = g

    def _render_layers(self, P, groups, ss, L, pal, haze, haze_k, haze_max, rim, under=0.5, yg=-1e9, decorate=False,
                       bark_bias=0.35, bark=BARK, prune_m=0.25, fl_own=0.06, prune_rz=0.025, entry=True,
                       sky_fill=(0.4, 0.56, 1.0), bounce=(0.98, 0.52, 0.62), mass=None, tag=''):
        W, H = self.W, self.H
        f = self.cam.f
        Wc = W + 2 * self.tmx
        Hs, Ws = H * ss, Wc * ss
        # per-tree form ellipses from the big clumps
        ntr = int(P[:, R.K_TREE].max()) + 1
        TF = np.zeros((ntr, 4), np.float64)
        for t in range(ntr):
            sel = (P[:, R.K_TREE] == t) & (P[:, R.K_KIND] == 0) & (P[:, R.K_PAR] < 0) & (P[:, R.K_R0] > 0)
            if not sel.any():
                TF[t] = (0, 0, 1, 1)
                continue
            x = P[sel, R.K_X0]
            y = P[sel, R.K_Y0]
            r = P[sel, R.K_R0]
            x0, x1 = (x - r).min(), (x + r).max()
            y0, y1 = (y - r).min(), (y + r).max()
            TF[t] = ((x0 + x1) / 2, (y0 + y1) / 2 + 0.1 * (y1 - y0), (x1 - x0) / 2 * 1.1, (y1 - y0) / 2 * 1.15)
        sky_fill = np.array(sky_fill, np.float32)
        bounce = np.array(bounce, np.float32)
        glow = np.array([0.62, 0.46, 0.38], np.float32)
        bins = []
        for g in groups:
            sel = g['sel'] & ((P[:, R.K_R0] > 0.35) | (P[:, R.K_R1] > 0.35))
            if not sel.any():
                continue
            hero = g['hero']
            zc = g['z']
            Pb = P[sel].copy()
            kind = Pb[:, R.K_KIND]
            rr = np.maximum(Pb[:, R.K_R0], Pb[:, R.K_R1])
            xmin = np.where(kind == 0, Pb[:, R.K_X0], np.minimum(Pb[:, R.K_X0], Pb[:, R.K_X1])) - rr
            xmax = np.where(kind == 0, Pb[:, R.K_X0], np.maximum(Pb[:, R.K_X0], Pb[:, R.K_X1])) + rr
            ymin = np.where(kind == 0, Pb[:, R.K_Y0], np.minimum(Pb[:, R.K_Y0], Pb[:, R.K_Y1])) - rr
            ymax = np.where(kind == 0, Pb[:, R.K_Y0], np.maximum(Pb[:, R.K_Y0], Pb[:, R.K_Y1])) + rr
            X0 = int(max(0, math.floor(xmin.min() / ss) * ss - 4))
            X1 = int(min(Ws, math.ceil(xmax.max() / ss) * ss + 4))
            Y0 = int(max(0, math.floor(ymin.min() / ss) * ss - 4))
            Y1 = int(min(Hs, math.ceil(ymax.max() / ss) * ss + 4))
            if X1 <= X0 or Y1 <= Y0:
                continue
            X0 -= X0 % ss
            Y0 -= Y0 % ss
            X1 += (-X1) % ss
            Y1 += (-Y1) % ss
            h, w = Y1 - Y0, X1 - X0
            idx = np.nonzero(sel)[0]
            Psub = P[idx].copy()
            for kx in (R.K_X0, R.K_X1):
                Psub[:, kx] -= X0
            for ky in (R.K_Y0, R.K_Y1):
                Psub[:, ky] -= Y0
            def zbuf(Ps):
                flat, ptr = R.build_bands(Ps, h, 32)
                zb_ = np.full((h, w), 1e9, np.float32)
                ib_ = np.full((h, w), -1, np.int32)
                R.zbuffer(h, w, Ps, flat, ptr, 32, zb_, ib_, float(self.cam.hy * ss), float(f * ss), float(Y0),
                          float(yg), float(bark_bias))
                return zb_, ib_
            zb, ibl = zbuf(Psub)
            # remove bark that only shows as small disconnected slivers between the blossom masses
            amin = (f * ss * prune_m / zc) ** 2
            kill = CN.prune_fragments(ibl, Psub, amin, rz_max=prune_rz)
            if len(kill):
                keep = np.ones(len(idx), bool)
                keep[kill] = False
                idx = idx[keep]
                Psub = Psub[keep]
                zb, ibl = zbuf(Psub)
            ib = np.where(ibl >= 0, idx[np.maximum(ibl, 0)], -1).astype(np.int32)
            m = (ib >= 0).astype(np.float32)
            sig = float(np.clip(f * ss * 0.3 / zc, 1.5, 80))
            # AO from a z-buffer smoothed at floret scale: only clump-scale crevices darken (no ring around
            # every little floret -> no 'bubble wrap')
            sig0 = float(np.clip(f * ss * 0.09 / zc, 0.8, 30))
            zb = zb.copy()
            zraw = zb
            zf0 = np.where(ib >= 0, zb, 0).astype(np.float32)
            zbs = F.fast_blur(zf0, sig0) / np.maximum(F.fast_blur(m, sig0), 1e-3)
            zb = np.where(ib >= 0, zbs, zb).astype(np.float32)
            zf = np.where(ib >= 0, zb, 0).astype(np.float32)
            bl = F.fast_blur(zf, sig) / np.maximum(F.fast_blur(m, sig), 1e-3)
            bl2 = F.fast_blur(zf, sig * 3) / np.maximum(F.fast_blur(m, sig * 3), 1e-3)
            zsafe = np.maximum(zb, 0.5)
            ao = np.clip((zb - bl) * (f * ss / zsafe) / sig * 1.2, 0, 1) * 0.6 + \
                np.clip((zb - bl2) * (f * ss / zsafe) / (sig * 3) * 1.2, 0, 1) * 0.4
            ao = (ao * m).astype(np.float32)
            zb = zraw
            out = np.zeros((h, w, 4), np.float32)
            edge = R.lobe_lines(zb, ib, P, 2, 0.004)
            palb = PAL_HERO if hero else pal
            # outer silhouette of the wood on the sun side (for a clean 2-3 px warm rim)
            barkm = (np.where(ib >= 0, P[np.maximum(ib, 0), R.K_MAT], -1) == 2).astype(np.float32)
            wr = max(2.0, 0.0014 * W * ss)
            # screen direction from this card toward the sun
            bcx = (X0 + X1) / (2 * ss) - self.tmx
            bcy = (Y0 + Y1) / (2 * ss)
            sdx, sdy = self.sun_xy[0] - bcx, self.sun_xy[1] - bcy
            sdl = math.hypot(sdx, sdy) + 1e-6
            lxs, lys = sdx / sdl, sdy / sdl
            sh_ = CN.shift_rep(barkm, -lys * wr, -lxs * wr)
            bedge = np.clip(barkm - sh_, 0, 1)
            bedge = cv2.GaussianBlur(bedge, (0, 0), max(0.6, wr * 0.25)) * barkm
            bedge = np.clip(bedge * 1.6, 0, 1).astype(np.float32)
            bD = cv2.distanceTransform((barkm > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            bDs = cv2.GaussianBlur(bD, (0, 0), 1.2)
            bgx = -cv2.Sobel(bDs, cv2.CV_32F, 1, 0, ksize=3)
            bgy = -cv2.Sobel(bDs, cv2.CV_32F, 0, 1, ksize=3)
            gn = np.sqrt(bgx * bgx + bgy * bgy) + 1e-6
            bgx = (bgx / gn).astype(np.float32)
            bgy = (bgy / gn).astype(np.float32)
            bD = np.where(barkm > 0, np.maximum(bD - 0.5, 0), -1).astype(np.float32)
            R.shade(zb, ib, ao, edge, P, TF, L, palb, bark, sky_fill, bounce, haze, haze_k, haze_max, glow, rim, out,
                    float(X0), float(Y0), 0.22, 0.45 if hero else 0.0, float(self.cam.hy * ss),
                    float(f * ss), under, 0.04 if hero else fl_own, bedge, bD, bgx, bgy)
            if os.environ.get('S05_DUMP'):
                np.savez_compressed(os.path.join(os.environ['S05_DUMP'], f'{tag}_{len(bins)}.npz'), out=out, zb=zb,
                                    ib=ib, off=np.array([X0, Y0]), zc=zc, fss=f * ss, hy=self.cam.hy * ss)
            if mass is not None:
                self._repaint(out, zb, ib, P, mass, tag, zc, f * ss, (float(X0), float(Y0)), haze, haze_k, haze_max,
                              len(bins), (lxs, lys))
            if decorate:
                self._blossom_breakup(out, ib, zb, P, zc, f * ss, len(bins))
            if entry:
                hk = haze_max * (1.0 - math.exp(-zc / haze_k))
                CN.entry_florets(out, zb, ib, P, len(bins), max(1.8, f * ss * 0.036 / zc), prims=idx,
                                 off=(float(X0), float(Y0)), bark=bark, haze=(haze, hk),
                                 light_dir=(lxs, lys), rz_trunk=0.14)
            pm = out[..., :3] * out[..., 3:4]
            sz = (w // ss, h // ss)
            a = cv2.resize(out[..., 3], sz, interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(pm, sz, interpolation=cv2.INTER_AREA) / np.maximum(a, 1e-4)[..., None]
            rgba = np.dstack([rgb, a]).astype(np.float32)
            bins.append(dict(rgba=rgba, ox=X0 // ss - self.tmx, oy=Y0 // ss, z=zc, hero=hero))
        return bins

    def _repaint(self, out, zb, ib, P, mass, tag, zc, fss, off, haze, haze_k, haze_max, seed, ldir):
        near = tag == 'near'
        pal = MASS_NEAR if near else MASS_FAR
        L = (-0.55, 0.72, 0.42) if near else (-0.5, 0.7, 0.1)
        if not near:
            haze_max = haze_max * 0.45
        ki = MS.repaint(out, zb, ib, P, mass, pal, L, fss, off, haze, haze_k, haze_max, self.cam.hy * fss / self.cam.f,
                        seed=seed, zc=zc, glow=GLOW_NEAR if near else GLOW_FAR, trans=1.0 if near else 0.8)
        hk = haze_max * (1.0 - math.exp(-zc / haze_k))
        lac = MS.lace(out, ib, P, mass, fss, zc, pal, seed=seed + 50, ki=ki, hazek=hk, haze=haze,
                      fsz=0.05 if near else 0.09, twig=None,
                      holes=1.0 if near else 0.5)
        out[:] = lac

    @staticmethod
    def _blossom_breakup(out, ib, zb, P, zc, fss, seed):
        """Where a branch disappears BEHIND a blossom mass, let the blossom spill over the wood as a short
        lobed fringe of flower clusters, so the branch fades into the mass instead of ending in a hard cut."""
        valid = ib >= 0
        mat = np.where(valid, P[np.maximum(ib, 0), R.K_MAT], -1)
        bark = mat == 2
        blos = (mat == 0) | (mat == 1)
        if not bark.any() or not blos.any():
            return
        wpx = float(np.clip(fss * 0.08 / zc, 3.0, 40.0))
        d = cv2.distanceTransform((~blos).astype(np.uint8), cv2.DIST_L2, 5)
        near = bark & (d < wpx)
        if not near.any():
            return
        # only where the adjacent blossom is in front of the wood
        k = int(wpx * 2) | 1
        zbl = np.where(blos, zb, 1e9).astype(np.float32)
        zmin = cv2.erode(zbl, np.ones((k, k), np.uint8))
        front = zmin < zb - 0.05
        h, w = ib.shape
        rng = np.random.default_rng(900 + seed)
        cell = max(2.0, wpx * 0.45)
        nh, nw = int(h / cell) + 2, int(w / cell) + 2
        nz = cv2.resize(rng.random((nh, nw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        reach = wpx * (0.15 + 0.85 * np.clip(nz, 0, 1) ** 2)
        zone = (near & front).astype(np.float32) * np.clip((reach - d) / 1.2, 0, 1)
        sig = max(1.0, wpx * 0.5)
        bm = blos.astype(np.float32)
        col = cv2.GaussianBlur(out[..., :3] * bm[..., None], (0, 0), sig) /             np.maximum(cv2.GaussianBlur(bm, (0, 0), sig), 1e-3)[..., None]
        out[..., :3] += (col - out[..., :3]) * zone[..., None]

    # ================================================================== reflection
    def _build_reflection(self):
        W, H = self.W, self.H
        Wc = W + 2 * self.emx
        img = np.zeros((H, Wc, 3), np.float32)
        sky = S.drift(self.sky, Wc, H, 0.0, 0.0)
        img[:] = sky
        for L in self.clouds.render_layers(Wc, H, 0.0):
            img = F.over_rgba(img, L)
        img = img * (1 - self.far_a[..., None] + self.water_a[..., None]) + self.far_pm
        for bn in self.bins_far:
            E.over_shift(img, bn['rgba'], bn['ox'] + self.emx, bn['oy'], 0.0)
        E.over_shift(img, self.overlay, 0, 0, 0.0)
        # soften: water is never a perfect mirror
        # vertical streaking (reflections stretch down), little horizontal blur so trunks / buildings stay legible
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=max(0.6, 0.0008 * W), sigmaY=max(1.0, 0.004 * W))
        self.refl = img.astype(np.float32)

    # ================================================================== petals
    def _init_petals(self):
        rng = np.random.default_rng(9)
        n = 700
        # depth distribution: many mid, some near (defocused), some far specks
        z = np.concatenate([rng.uniform(1.1, 1.7, 4), rng.uniform(1.7, 3.5, 70), rng.uniform(3.5, 10, 260),
                            rng.uniform(10, 30, 260)])
        n = len(z)
        self.p_z = z
        self.p_u = rng.random(n)
        self.p_v = rng.random(n)
        self.p_size = rng.uniform(0.02, 0.03, n)
        self.p_size[:4] = rng.uniform(0.022, 0.028, 4)
        self.p_ph = rng.uniform(0, 6.283, n)
        self.p_spin = rng.uniform(1.5, 4.0, n) * rng.choice([-1, 1], n)
        self.p_fall = rng.uniform(0.35, 0.7, n)
        self.p_wind = rng.uniform(0.35, 0.8, n)
        self.p_tone = rng.uniform(0, 1, n)
        # gust group (round 7 accent): mid-depth petals the wind lifts into the sky during the tilt-up
        self.p_gust = ((z > 1.8) & (z < 12.0) & (rng.random(n) < 0.45)).astype(np.float64)

    def _petals(self, img, t, tx, D=0.0):
        W, H = self.W, self.H
        f = self.cam.f
        z = self.p_z
        # tilt-up: the petals are in the world, so they slide down with the scene by D px ... except the
        # gust group, which is lifted by the wind and rises with the camera (the move follows them upward)
        c = self.crane(t) / max(CRANE * H, 1e-6)                       # 0..1 progress of the accent
        lift = self.p_gust * (0.92 * D + 0.06 * H * c) / f * z          # metres of upward lift
        tilt = D / f * z
        # visible half-extent at depth z (metres) + margin, used for wrapping
        half_w = (W * 0.62) / f * z
        half_h = (H * 0.62) / f * z
        X = (self.p_u * 2 - 1) * half_w - self.p_wind * t + 0.05 * np.sin(t * 1.3 + self.p_ph) * z ** 0.3
        Y = (self.p_v * 2 - 1) * half_h - self.p_fall * t + 0.06 * np.sin(t * 2.1 + self.p_ph * 1.7)
        Y = Y - tilt + lift
        X = X - self.p_gust * 0.35 * c * c * z * 0.5
        # camera parallax + wrap inside the (depth-dependent) window
        Xs = X - tx
        X = np.mod(Xs + half_w, 2 * half_w) - half_w
        Y = np.mod(Y + half_h, 2 * half_h) - half_h
        xs = self.cam.cx + f * X / z
        ys = self.cam.hy - f * Y / z
        spin = t * self.p_spin + self.p_ph
        r = f * self.p_size / z
        ax = r * 0.62
        bx = r * 0.42 * (0.25 + 0.75 * np.abs(np.cos(spin * 0.7 + self.p_ph)))
        ang = spin * 0.5 + self.p_ph
        coc = np.abs(1.0 / z - 1.0 / 7.0) * 0.0022 * f
        coc = np.clip(coc - 0.6, 0, 0.0012 * W)
        face = np.abs(np.cos(spin * 0.7 + self.p_ph))
        lit = 0.5 + 0.5 * face
        col = np.stack([0.97 + 0.05 * lit, 0.68 + 0.16 * lit + 0.05 * self.p_tone, 0.8 + 0.1 * lit], 1)
        # far petals blend toward the haze
        hz = np.clip((z - 10) / 60, 0, 0.5)[:, None]
        col = col * (1 - hz) + np.array([0.9, 0.88, 0.95]) * hz
        near = z < 1.75
        col[near] = np.stack([1.0 + 0.06 * lit[near], 0.6 + 0.24 * lit[near], 0.74 + 0.14 * lit[near]], 1)
        bx[near] = np.maximum(bx[near], ax[near] * 0.35)
        coc[near] = np.minimum(coc[near], 0.0006 * W)
        alpha = np.clip(0.95 - 0.3 * hz[:, 0], 0, 1)
        alpha[near] = 1.0
        order = np.argsort(-z)
        R.draw_petals(img, xs[order].astype(np.float64), ys[order].astype(np.float64), ax[order].astype(np.float64),
                      bx[order].astype(np.float64), ang[order].astype(np.float64), coc[order].astype(np.float64),
                      col[order].astype(np.float64), alpha[order].astype(np.float64))
        return img

    # ================================================================== railing
    def _railing(self, img, tx):
        cam = self.cam
        f = cam.f
        caps = []
        u = E.U_RAIL
        s_lo, s_hi = 1.0, 128.0
        posts = np.arange(0.4, s_hi, 2.0)
        # rails: split at posts (straight segments; 1/Z is affine along a line in screen space)
        segs = []
        for hr, rad, spk in ((1.05, 0.03, 1.9), (0.55, 0.02, 0.55)):
            ss_ = np.concatenate([[s_lo], posts[posts > s_lo]])
            for a, b in zip(ss_[:-1], ss_[1:]):
                segs.append((u, E.Y_PATH + hr, a, u, E.Y_PATH + hr, b, rad, spk))
        for p in posts:
            segs.append((u, E.Y_PATH + 0.05, p, u, E.Y_PATH + 1.08, p, 0.028, 0.35))
        S_ = np.array(segs)
        X0, Y0, Z0 = cam.to_cam(S_[:, 0], S_[:, 1], S_[:, 2])
        X1, Y1, Z1 = cam.to_cam(S_[:, 3], S_[:, 4], S_[:, 5])
        # clip against the near plane
        zn = 0.8
        ok = (Z0 > zn) | (Z1 > zn)
        S_, X0, Y0, Z0, X1, Y1, Z1 = S_[ok], X0[ok], Y0[ok], Z0[ok], X1[ok], Y1[ok], Z1[ok]
        tclip = np.clip((zn - Z0) / (Z1 - Z0 + 1e-9), 0, 1)
        m0 = Z0 < zn
        X0 = np.where(m0, X0 + (X1 - X0) * tclip, X0)
        Y0 = np.where(m0, Y0 + (Y1 - Y0) * tclip, Y0)
        Z0 = np.where(m0, zn, Z0)
        x0, y0 = cam.proj(X0, Y0, Z0, tx)
        x1, y1 = cam.proj(X1, Y1, Z1, tx)
        r0 = f * S_[:, 6] / Z0
        r1 = f * S_[:, 6] / Z1
        zm = 0.5 * (Z0 + Z1)
        order = np.argsort(-zm)
        sid = np.arange(len(S_), dtype=np.float64)
        Cp = np.stack([x0, y0, r0, x1, y1, r1, S_[:, 7], sid], 1)[order].astype(np.float64)
        RL.draw_pipes(img, Cp, np.array([0.3, 0.36, 0.58]), np.array([0.8, 0.82, 0.9]),
                      np.array([1.15, 0.88, 0.6]), np.array([1.25, 1.1, 0.85]), np.array([0.74, 0.6, 0.55]),
                      float(tx * 0.35))
        return img

    # ================================================================== frame
    def cam_x(self, t):
        return TRAVEL * (_ease(t / DURATION) - 0.5)

    def frame(self, t):
        W, H = self.W, self.H
        cam = self.cam
        f = cam.f
        tx = self.cam_x(t)
        M = self.M
        D = self.crane(t)
        # sky + clouds (effectively at infinity; a hint of parallax for depth), on a canvas with M rows of
        # headroom above the frame; everything below the frame top is painted into the view `img`
        canvas = self._sky(t, tx)
        img = canvas[M:]
        # near-bank embankment wall (only seen through the gap that opens between the path edge and the water
        # plate at the ends of the dolly)
        r0 = int(self.cam.hy) + 4
        img[r0:] = self._wall[r0:]
        sun_x, sun_y = self.sun_xy
        # far plate with water
        buf = self.far_pm.copy()
        E.water_kernel(buf, self.w_y, self.w_x, self.w_u, self.w_s, self.w_Z, self.w_a, self.w_ry, self.refl,
                       self.ywb, float(t), float(f), float(sun_x + self.emx), float(2 * cam.hy - sun_y),
                       np.array([0.02, 0.15, 0.4], np.float32), np.array([0.7, 0.55, 0.37], np.float32),
                       self.far_invz, float(f * tx))
        far = self._far_rgba
        np.multiply(buf, self._far_ia, out=far[..., :3])
        E.over_shift_map(img, far, -self.emx, 0, self.far_invz, float(f * tx))
        for bn in sorted(self.bins_far, key=lambda b: -b['z']):
            E.over_shift(img, bn['rgba'], bn['ox'], bn['oy'], -f * tx / bn['z'])
        E.over_shift_map(img, self.overlay, -self.emx, 0, self.far_invz, float(f * tx))
        self._cards(img, tx, 0)
        self._water_sparkles(img, t, tx)
        # near ground: exact per-row shear of a horizontal plane
        rows = np.arange(H, dtype=np.float64) + 0.5
        shifts = -tx * np.maximum(rows - cam.hy, 0) / E.EYE
        E.over_shift_rows(img, self.ground, -self.emx, 0, shifts)
        self.dapple.apply(img, t, shifts)
        self._cards(img, tx, 1)
        G.composite(img, self.grass, self.emx, lambda r: shifts[min(max(int(r), 0), H - 1)])
        img = self._railing(img, tx)
        self._rail_sparkles(img, t, tx)
        occ_c = np.zeros((H + M, W), np.float32)
        gsh = lambda r: shifts[min(max(int(r), 0), H - 1)]
        for bn in self.bins_near_sorted:
            MO.over_rows_acc(canvas, occ_c, bn['rgba'], bn['ox'], bn['oy'] + M, -f * tx / bn['z'],
                             MO.sway_px(t, bn['z'], W, bn['phase']), bn['foot'], 0.0)
            gf = self.grass_feet.get(round(float(bn['z']), 4))
            if gf:
                G.composite(img, gf, self.emx, gsh)
        # ---- hand-painted foreground branch (nearest plane: largest parallax, gentle sway)
        self._hero(canvas, occ_c, t, tx)
        # ---- tilt-up: crop the frame out of the canvas (sub-pixel, eased)
        img, occ, shifts = self._crop(canvas, occ_c, D, shifts)
        sun_y = sun_y + D
        vis = F.sun_visibility(occ, sun_x, sun_y, 0.012 * W)
        # ---- light beams slanting from the canopy gaps down onto the path (sun-ray vanishing point = sun)
        #      + crepuscular shafts from the sun through the branch / canopy gaps (one low-res pass)
        b, sh, c1, c2 = self._beams(shifts, sun_x, sun_y, occ, D)
        lo = b[..., None] * (0.55 * c1) + sh[..., None] * ((0.5 + 0.35 * vis) * c2)
        img += cv2.resize(lo, (W, H), interpolation=cv2.INTER_LINEAR)
        img = self._petals(img, t, tx, D)
        img = SFX.bloom_fast(img, threshold=0.93, knee=0.28, strength=0.26, halation=0.1)
        img += self.flare.render(sun_x, sun_y, vis=0.35 + 0.65 * vis, rot=0.02 * t,
                                      center=(W / 2 - tx * 0.08 * W, H / 2))
        img = F.shoulder(img, 0.82, desat=0.2)
        img *= self.paper
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.0, vig=0.2, ca=0.0005)

    def crane(self, t):
        u = np.clip((t - CRANE_T[0]) / (CRANE_T[1] - CRANE_T[0]), 0.0, 1.0)
        # round 11: symmetric blend of linear + sine ease: the move is already drifting on the cut in and still
        # drifting into the dissolve, with a peak speed only ~1.3x the mean (no acceleration spike)
        u = 0.45 * float(u) + 0.55 * (0.5 - 0.5 * math.cos(math.pi * float(u)))
        return float(CRANE * self.H * u)

    def _crop(self, canvas, occ_c, D, shifts):
        W, H, M = self.W, self.H, self.M
        if D < 1e-3:
            return np.ascontiguousarray(canvas[M:]), np.ascontiguousarray(occ_c[M:]), shifts
        A = np.float32([[1, 0, 0], [0, 1, D - M]])
        img = cv2.warpAffine(canvas, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        occ = cv2.warpAffine(occ_c, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img, occ, shifts

    def _paper(self):
        """Static paint / paper texture (multiplier ~1 +- 2%): coarse wash blotches, fine paper tooth and
        faint diagonal brush fibres.  Fixed to the frame (like a painted cel), never animated -> no flicker."""
        W, H = self.W, self.H
        rng = np.random.default_rng(1234)
        u = W / 1920.0

        def nz(cell):
            gh, gw = int(H / cell) + 3, int(W / cell) + 3
            g = rng.standard_normal((gh, gw)).astype(np.float32)
            return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:H, :W]
        p = 0.006 * nz(90 * u) + 0.004 * nz(22 * u) + 0.005 * nz(max(1.5, 2.2 * u))
        fib = rng.standard_normal((H, W)).astype(np.float32)
        k = max(3, int(9 * u) | 1)
        ker = np.eye(k, dtype=np.float32)[::-1] / k
        fib = cv2.filter2D(fib, -1, ker) * 0.006
        p = 1.0 + p + fib
        return np.clip(p, 0.94, 1.06)[..., None].astype(np.float32)

    def _cards(self, img, tx, which):
        pm, r0, invz = self.cards[which]
        f = self.cam.f
        W = self.W
        Wc = pm.shape[1]
        j = np.arange(Wc, dtype=np.float64)
        d = j - self.emx - f * tx * invz            # destination column of each source column
        src = np.interp(np.arange(W, dtype=np.float64), d, j, left=-10, right=-10).astype(np.float32)
        h = pm.shape[0]
        mapx = np.broadcast_to(src[None, :], (h, W))
        mapy = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, W))
        c = cv2.remap(pm, np.ascontiguousarray(mapx), np.ascontiguousarray(mapy), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
        sub = img[r0:r0 + h]
        sub *= (1 - c[..., 3:4])
        sub += c[..., :3]

    def _hero(self, img, occ, t, tx):
        W, H = self.W, self.H
        f = self.cam.f
        sh = -f * tx / HERO_Z + W * 0.004 * math.sin(t * 0.9 + 0.5)
        ang = 1.1 * math.sin(t * 1.05) + 0.45 * math.sin(t * 2.3 + 1.0)       # degrees, pivot at the edge
        piv = (0.0 + sh, 0.03 * H + self.M)
        A = cv2.getRotationMatrix2D(piv, ang, 1.0)
        Tm = np.array([[1, 0, self.hero_ox + sh], [0, 1, self.hero_oy + self.M], [0, 0, 1]], np.float64)
        M = (np.vstack([A, [0, 0, 1]]) @ Tm)[:2].astype(np.float32)
        Hr = self._hero_rows + self.M
        pm = cv2.warpAffine(self.hero_pm, M, (W, Hr), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = pm[..., 3]
        sub = img[:Hr]
        sub *= (1 - a)[..., None]
        sub += pm[..., :3]
        occ[:Hr] = occ[:Hr] * (1 - a) + a
        return img, occ

    def _beams(self, shifts, sx, sy, occ, D=0.0):
        W, H = self.W, self.H
        q = 4
        w, h = W // q, H // q
        ys = (np.arange(h) + 0.5) * q
        sh = np.interp(ys - D, np.arange(H) + 0.5, shifts)
        mx = ((np.arange(w)[None, :] + 0.5) * q + self.emx - sh[:, None]) / q
        spot_q = self._spot_q
        mapx = (mx - 0.5).astype(np.float32)
        mapy = np.repeat((np.arange(h, dtype=np.float32))[:, None], w, 1)
        src = cv2.remap(spot_q, mapx, (mapy - D / q).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        b = SFX.beams_toward(np.clip(src, 0, None) ** 2.2, sx / q, sy / q, length=0.5, steps=22)
        # sun shafts through the gaps of the branch / canopy, same low-res pass
        occ_q = np.clip(cv2.resize(occ, (w, h), interpolation=cv2.INTER_AREA), 0.0, 1.0)
        sh = SFX.sun_shafts(occ_q, sx / q, sy / q, strength=0.26, length=0.85, radius=0.07)
        c1 = np.array([1.0, 0.86, 0.66], np.float32)
        c2 = np.array([1.0, 0.9, 0.78], np.float32)
        return b, sh, c1, c2

    def _sky(self, t, tx):
        """sky + drifting clouds, rendered only down to just below the horizon (the rest is always covered)"""
        W, H = self.W, self.H
        f = self.cam.f
        M = self.M
        Hs = self._sky_rows + M                     # canvas rows (M rows of headroom above the frame)
        oy = -(H + M - Hs) / 2                      # canvas row 0 <-> plate row 0
        img = S.drift(self.sky, W, Hs, t, 0.0, cam=(tx * 40, 0), depth=0.05, offset=(0, oy))
        for p_, sp, dp, bl, i in self.clouds.layers:
            L = S.drift(p_, W, Hs, t, sp, (tx * f * 0.02, 0.0), 1.0, dp, offset=(0, oy), billow=bl, seed=31 * i)
            img = F.over_rgba(img, L)
        out = np.empty((H + M, W, 3), np.float32)
        out[:Hs] = img
        out[Hs:] = img[-1:]
        return out

    def _rail_sparkles(self, img, t, tx):
        """sun glints along the top rail and the post caps (specular highlight follows the camera)"""
        cam = self.cam
        ss_ = self._rail_s
        X, Y, Z = cam.to_cam(E.U_RAIL + 0 * ss_, E.Y_PATH + 1.08, ss_ + 1.2 * tx)
        x, y = cam.proj(X, Y, Z, tx)
        k = np.clip(3.5 / Z, 0.08, 1.0)
        m = self._rail_keep
        SFX.sparkles(self.W, self.H, x[m], y[m], size=self.W * (0.005 + 0.012 * k[m]) * self._rail_sz[m],
                     inten=1.6 * k[m] ** 0.5 * self._rail_in[m], t=t, seed=5, out=img)

    def _water_sparkles(self, img, t, tx):
        f = self.cam.f
        W, H = self.W, self.H
        # warm sheen column, moved with the water plate's parallax (per-row shift)
        shr = (-f * tx * self._sheen_invz).astype(np.float32)
        mapx = (np.arange(W, dtype=np.float32)[None, :] - shr[:, None]).astype(np.float32)
        mapy = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, 1)
        img += cv2.remap(self._sheen, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        g = self.glit
        self.glit.draw(img, t, -f * tx * self.far_invz[g.py, g.px].astype(np.float64))
        # a few faint sky glints far out on the water (only the far half of the list)
        x = self._ws_x - self.emx - f * tx * self.far_invz[self._ws_py, self._ws_px]
        n1 = 34
        # round 9: clusters of bright star glints in the sun's glitter path (sizes fixed, gentle twinkle)
        SFX.sparkles(W, H, x[:n1], self._ws_y[:n1], size=self._ws_sz[:n1] * 0.8, inten=self._ws_in[:n1] * 1.5,
                     t=t, seed=7, out=img)
        SFX.sparkles(W, H, x[n1:], self._ws_y[n1:], size=self._ws_sz[n1:] * 0.6, inten=self._ws_in[n1:] * 0.6,
                     t=t, seed=9, out=img)
