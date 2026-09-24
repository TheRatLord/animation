"""Clouds round 3 final (lib/clouds3) demo: a towering summer cumulus over a distant sea (wwy_01 / gow_02 /
cm5_04), sun upper right, utility pole and wires in front.

Plates:
  * hero towering cumulus left of centre (volumetric painter vol_cloud_plate): stacked heads, tower-on-tower
    lobes shrinking toward the crown, warm cream sunlit flank / cool blue-violet shade flank from a real
    sun march, broad painted value planes, crisp cauliflower on the lit silhouette (edge_florets), lost
    soft edges on the shade side, a flat slightly darker base sitting in the horizon haze;
  * distant heaps on the right and a receding horizon bank (same painter, paler / lower contrast);
  * feathered stratus layers at the cloud bases, a thin aerial-haze band on the horizon;
  * mackerel sky (sky_rows_plate): cloudlet streets on a perspective sky plane, lit sun-side edges,
    gathered in patches in the upper right, compressing into streaks toward the horizon; faint cirrus;
  * sea: deep cobalt foreground to a pale horizon line, specular glitter path under the sun;
  * foreground utility pole + swaying wires; Shinkai flare, bloom.
Camera: eased push-in + pan left with multi-plane parallax (sky 0.04 .. bank 0.22 .. tower 0.35 ..
pole 1.8), crane on the pole only; billowing cloud edges, drifting cloudlets, wire sway, sea twinkle.
"""
import time
import math
import numpy as np

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds3 as K

DURATION = 8.0

SKY = dict(stops=[(0.0, '#06237e'), (0.18, '#0b36a0'), (0.4, '#1a5cc4'), (0.62, '#3c8fdc'),
                  (0.8, '#7ec3ef'), (0.92, '#b9e2f6'), (1.0, '#dcf1fa')],
           sun_glow='#dff6ff', sun_glow_amt=0.34, below='#dcf1fa', band=('#eefafd', 0.55))
SUN_DIR = (0.75, -0.65)


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        sc = W / 1920.0
        m = 0.09
        PW, PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        self.PW, self.PH = PW, PH
        ox, oy = W * m, H * m
        self.ox, self.oy = ox, oy
        self.hz = 0.9
        hy = self.hz * H + oy
        self.sun = (0.84 * W, 0.1 * H)
        self.sky = S.sky_gradient(W, H, SKY, horizon=self.hz, sun=self.sun, sun_radius=0.75, variation=0.015, seed=2)
        by = hy - 0.004 * H
        # hero towering cumulus
        self.tower = K.vol_cloud_plate(PW, PH, [dict(cx=0.33 * W + ox, base_y=by, width=0.7 * H, height=0.86 * H,
                                                     seed=41, lean=0.08)], sun_dir=SUN_DIR, sun_z=-0.1,
                                       max_dim=260, paint=dict(base_y=by, base_dark=0.12, base_h=0.06), seed=12)
        # distant heaps on the right
        self.heap = K.vol_cloud_plate(PW, PH, [dict(cx=0.8 * W + ox, base_y=by + 0.001 * H, width=0.36 * W,
                                                    height=0.25 * H, seed=21, kind='heap'),
                                               dict(cx=0.99 * W + ox, base_y=by + 0.002 * H, width=0.2 * W,
                                                    height=0.14 * H, seed=33, kind='heap', dz=200 * sc)],
                                      sun_dir=SUN_DIR, sun_z=-0.1, max_dim=260, haze=0.2,
                                      paint=dict(base_y=by, base_dark=0.1, base_h=0.04), seed=5)
        # low receding horizon bank (pale, low contrast)
        rng = np.random.default_rng(7)
        bank = []
        x = -0.1 * W
        while x < 1.2 * W:
            wd = W * rng.uniform(0.04, 0.1)
            bank.append(dict(cx=x + ox, base_y=hy - 0.0005 * H, height=wd * rng.uniform(0.22, 0.42), width=wd,
                             kind='heap', seed=int(rng.integers(1 << 30)), sizes=(0.16, 0.08),
                             dz=rng.uniform(0, 1) * W * 0.4))
            x += wd * rng.uniform(0.5, 0.85)
        self.bank = K.vol_cloud_plate(PW, PH, bank, sun_dir=SUN_DIR, sun_z=-0.1, max_dim=440, persp=12.0,
                                      haze=0.45, haze_col=(0.8, 0.9, 0.97), paint=dict(kuwa=2), florets=False,
                                      seed=6)
        # feathered stratus layers at the cloud bases
        self.strat = K.strata_plate(PW, PH, hy - 0.022 * H, 0.006 * H, seed=12, count=7, length=(0.15, 0.4),
                                    opacity=0.7)
        self.strat2 = K.strata_plate(PW, PH, hy - 0.065 * H, 0.004 * H, seed=19, count=3, length=(0.12, 0.3),
                                     opacity=0.45, x_range=(0.45, 1.1))
        # cirrus + mackerel streets (upper right, clear of the tower)
        self.cirrus = K.cirrus_plate(PW, PH, preset='noon', seed=9, region=(0.02, 0.45), angle=-14, density=0.5,
                                     opacity=0.4)
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        avoid = np.exp(-(((xx - 0.33 * W - ox) / (0.3 * W)) ** 2 + ((yy - 0.45 * H - oy) / (0.7 * H)) ** 2) * 2.0)
        keep = K._ss(0.35 * W, 0.75 * W, xx - ox)
        self.flecks = K.sky_rows_plate(PW, PH, hy, seed=13, alt=8.0, cell=0.24, cell_stretch=1.5, row_l=0.7,
                                       patch=9.0, patch_thr=0.05, cover=0.45, warp=1.2, angle=-58.0,
                                       region=(0.0, 0.72), z_haze=90.0, lit=(1.1, 1.05, 0.94),
                                       body=(0.93, 0.96, 0.99), under=(0.7, 0.8, 0.95), a_lo=0.12, a_hi=0.3,
                                       lumps=0.5, lit_screen=(3.0, -3.0), opacity=1.0, haze_fade=0.5, veil=0.25,
                                       edge_px=1.0, avoid=np.clip(avoid + (1 - keep), 0, 1))
        self.irid = K.iridescence_plate(self.tower[..., 3], (0.56 * W + ox, 0.36 * H + oy), 0.09 * W, strength=0.04,
                                        band=(0.006, 0.07))
        # aerial haze band: the cloud bases sit in it
        self.haze = K.haze_plate(PW, PH, hy - 0.06 * H, hy + 0.001 * H, '#d9f0fa', amount=0.55, y2=hy + 0.025 * H)
        self.sea, self.glit = K.sea_plate(PW, PH, hy, self.sun[0] + ox, near='#082a74', far='#5f9fd4', path_w=0.045,
                                          seed=4, haze=('#e2f4fb', 0.75), glitter=1.6)
        # thin bright horizon line
        self.hline = np.zeros((PH, PW, 4), np.float32)
        ys = np.arange(PH, dtype=np.float32)[:, None]
        self.hline[..., :3] = np.array([0.93, 0.98, 1.0], np.float32)
        self.hline[..., 3] = np.exp(-((ys - hy - 0.5) / (1.2 * sc + 0.3)) ** 2) * 0.55 * np.ones((1, PW), np.float32)
        # foreground pole + wire anchors
        self.pole, self.anchors = K.utility_pole_plate(PW, PH, 0.87 * W + ox, 0.15 * H + oy, sun_dir=SUN_DIR)
        rngw = np.random.default_rng(3)
        self.wire_spec = []
        for i, (axp, ayp) in enumerate(self.anchors[:6]):
            self.wire_spec.append((axp, ayp, -0.14 * W, 0.42 * H + i * 0.022 * H + rngw.uniform(-0.01, 0.01) * H,
                                   0.05 * H * rngw.uniform(0.85, 1.2), (2.0 if i < 4 else 1.4) * sc,
                                   rngw.uniform(0, 6.28)))
        self.flare = K.SunFlare(W, H, tint=(1.0, 0.95, 0.86), star_len=0.1, bloom=0.65, ring=0.6, chain=1.0)
        self.setup_s = time.time() - t0

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.07 * p
        cam = (-0.05 * W * p, -0.012 * H * p)
        crane = -0.06 * H * p                          # the near pole drops through frame
        img = self.sky.copy()
        Lc = K.drift(self.cirrus, W, H, t, speed=0.0015, cam=cam, zoom=zoom, depth=0.04)
        Lk = K.drift(self.flecks, W, H, t, speed=(0.0035, -0.0008), cam=cam, zoom=zoom, depth=0.08)
        Li = K.drift(self.irid, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=0.35)
        Lb = K.drift(self.bank, W, H, t, speed=0.001, cam=cam, zoom=zoom, depth=0.22)
        Lh = K.drift(self.heap, W, H, t, speed=0.0018, cam=cam, zoom=zoom, depth=0.26, billow=0.0005,
                     billow_scale=0.02, billow_rate=0.3, seed=2)
        Lt = K.drift(self.tower, W, H, t, speed=0.0006, cam=cam, zoom=zoom, depth=0.35, billow=0.0007,
                     billow_scale=0.018, billow_rate=0.3, seed=1)
        Ls2 = K.drift(self.strat2, W, H, t, speed=0.0025, cam=cam, zoom=zoom, depth=0.3)
        Ls = K.drift(self.strat, W, H, t, speed=0.003, cam=cam, zoom=zoom, depth=0.36)
        Lhz = K.drift(self.haze, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=0.36)
        Lsea = K.drift(self.sea, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=0.4)
        Lln = K.drift(self.hline, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=0.4)
        img = K.composite(img, Lc, Lk)
        img = img + Li[..., :3] * Li[..., 3:4]
        img = K.composite(img, Lb, Ls2, Lh, Lt, Ls, Lhz, Lsea)
        img = img + Lln[..., :3] * Lln[..., 3:4]
        img = img + K.sea_glitter(self.glit, t, W, H, cam=cam, zoom=zoom, depth=0.4) * Lsea[..., 3:4]
        z0 = 1 + (zoom - 1) * 0.04
        lx = W / 2 + (self.sun[0] - W / 2) * z0 + cam[0] * 0.04
        ly = H / 2 + (self.sun[1] - H / 2) * z0 + cam[1] * 0.04
        img = img + S.sun(W, H, lx, ly, radius=0.014, color=(1.0, 0.97, 0.9), intensity=1.25, glow_size=0.3)
        occ = K.occluder(Lt, Lh)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.07, length=0.9, tint=(1.0, 0.96, 0.86),
                                   radius=0.16, t=t)
        pd = 1.6
        pcam = (cam[0], crane / pd)
        Lp = K.drift(self.pole, W, H, t, speed=0.0, cam=pcam, zoom=zoom, depth=pd)
        wires = []
        for (axp, ayp, x1, y1, sag, th, ph) in self.wire_spec:
            a0 = K.screen_pos((axp, ayp), W, H, (self.PW, self.PH), cam=pcam, zoom=zoom, depth=pd)
            a1 = K.screen_pos((x1 + self.ox, y1 + self.oy), W, H, (self.PW, self.PH), cam=pcam, zoom=zoom, depth=1.2)
            wires.append((a0[0], a0[1], a1[0], a1[1], sag, th, ph))
        Lw, glint = K.draw_wires(W, H, wires, t=t, sun=(lx, ly), glint=1.0, sway=0.035)
        img = K.composite(img, Lp, Lw)
        if glint is not None:
            img = img + glint
        vis = F.sun_visibility(K.occluder(Lp, Lw), lx, ly, 0.01 * W)
        shim = 1.0 + 0.04 * math.sin(1.7 * t)
        img = img + self.flare.render(lx, ly, intensity=0.9 * shim * (0.4 + 0.6 * vis), t=t,
                                      center=(W / 2 + cam[0] * 0.3, H / 2 + cam[1] * 0.3))
        img = F.bloom_soft(img, threshold=1.0, knee=0.2, strength=0.22, halation=0.12)
        img = F.shoulder(img, 0.9, 0.05)
        return F.finish_fast(img, t, sat=1.04, grain_amt=0.004, vig=0.2, ca=0.0006)
