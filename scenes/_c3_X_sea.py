"""Clouds round 3 final (lib/clouds3) demo: sea of clouds at sunrise seen from above (yn_02 / wwy_02).

Plates:
  * the deck (deck_fields + paint_deck): a ray-marched heightfield cloud volume in true perspective - big
    rolling banks near the camera, compressing into thin streaky bands with hard bright top lines at
    the horizon; crests facing the sun catch hot gold rims (brighter toward the sun column and the
    distance), bodies fall into navy-violet, the foreground is airbrushed soft; the two towers cast
    shadow wedges across the floor toward the camera;
  * two towers standing in the deck (vol_cloud_plate, side-lit from the low sun): a broad cauliflower
    congestus on the left and a farther anvil cumulonimbus on the right (flat top spreading downwind);
    warm peach crowns on the sun-facing lobes, cool violet bodies, lost soft edges on the shadow side,
    bases sunk into the deck and a mist skirt;
  * altocumulus streets in the upper sky on a perspective sky plane, lit gold on the edges facing the sun;
  * sky: navy zenith -> blue -> lilac -> peach, a thin white-hot horizon line; crepuscular rays;
  * Shinkai sun: hard small core on the horizon, anamorphic streak, faint halo ring, ghost-dot chain.
Camera: eased push-in + pan right. The deck is re-projected per frame from its depth map (true per-pixel
parallax: near banks slide faster than the towers and the horizon); towers occlude / are occluded by the
deck by depth; slow drift, billowing tower edges, rim shimmer.
"""
import math
import time
import numpy as np
import cv2

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds3 as K

DURATION = 8.0

SKY = dict(stops=[(0.0, '#0b1a55'), (0.3, '#1f3486'), (0.55, '#4a5aa8'), (0.72, '#8f86bf'),
                  (0.86, '#df9fa4'), (0.95, '#ffc08e'), (1.0, '#ffe3b4')],
           sun_glow='#ffd9a4', sun_glow_amt=0.3, below='#f3c3a4', band=('#fff0cf', 0.45))
FOV = 55.0


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        sc = W / 1920.0
        m = 0.07
        PW, PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        ox, oy = W * m, H * m
        self.PW, self.PH, self.ox, self.oy = PW, PH, ox, oy
        self.hz = 0.4
        hy = self.hz * H + oy
        self.hy = hy
        self.sun = (0.58 * W, 0.385 * H)
        sun_p = (self.sun[0] + ox, self.sun[1] + oy)
        self.sky = S.sky_gradient(W, H, SKY, horizon=self.hz, sun=self.sun, sun_radius=0.35, variation=0.01, seed=4,
                                  horizon_glow=0.8)
        f = (PW / 2.0) / math.tan(math.radians(FOV) / 2)
        self.f = f
        # tower placement (world: X right, Z forward, deck mean top at Y=0, camera at Y=1)
        towers = [dict(x=0.15 * W + ox, Z=13.0, hgt=1.9, wd=2.6, seed=5, anvil=None, lean=-0.04),
                  dict(x=0.8 * W + ox, Z=28.0, hgt=5.0, wd=2.2, seed=8, lean=0.05,
                       anvil=dict(left=2.0, right=6.0, thick=0.8))]
        cols = []
        for T in towers:
            T['X'] = (T['x'] - PW / 2.0) / f * T['Z']
            cols.append((T['X'], T['Z'], 0.36 * T['wd'], 0.2 + T['hgt']))
        # deck
        self.deck_R = K.deck_fields(PW, PH, hy, sun_p, fov=FOV, seed=3, towers=cols,
                                    stretch=2.2, bank=(0.45, 9.0, 3.2), p1=(0.36, 1.6), p3=(0.06, 0.18), p4=(0.012, 0.065))
        self.deck = K.paint_deck(self.deck_R, PW, PH, hy, sun_p[0], deep=(0.07, 0.08, 0.21), body=(0.3, 0.31, 0.53),
                                 near_dark=0.6, near_rim=0.08, near_z=(1.5, 9.0))
        x0, y0, x1, y1 = self.deck_R['box']
        self.deck_rgba = np.zeros((PH, PW, 4), np.float32)
        self.deck_rgba[y0:y1, x0:x1, :3] = self.deck
        self.deck_rgba[y0:y1, x0:x1, 3] = 1.0
        self.deck_rgba[:y0 + 3, :, 3] = 0.0
        self.deck_rgba[..., :3] = K._bleed(self.deck_rgba[..., :3], self.deck_rgba[..., 3], 3.0)
        Zm = np.full((PH, PW), 200.0, np.float32)
        Zm[y0:y1, x0:x1] = np.where(self.deck_R['A'] > 0.3, self.deck_R['Z'], 200.0)
        self.deckZ = Zm
        # towers
        self.towers = []
        for T in towers:
            ppu = f / T['Z']
            by = hy + (1.0 - 0.15) * ppu
            cyc = by - 0.6 * T['hgt'] * ppu
            L3 = (sun_p[0] - T['x'], sun_p[1] - cyc - 0.3 * H, 0.05 * W)
            an = None
            if T['anvil'] is not None:
                a = T['anvil']
                an = dict(left=a['left'] * ppu, right=a['right'] * ppu, thick=a['thick'] * ppu,
                          col_hw=0.45 * T['wd'] * ppu)
            P = K.vol_cloud_plate(PW, PH, [dict(cx=T['x'], base_y=by, width=T['wd'] * ppu, height=T['hgt'] * ppu,
                                                seed=T['seed'], lean=T['lean'], anvil=an)],
                                  L3=L3, eye_y=hy, persp=f / (T['hgt'] * ppu), max_dim=260,
                                  paint=dict(ramp='sunrise', base_y=by, k_sun=0.7, k_amb=0.25, lost=(0.0, 1.0)),
                                  florets=dict(sun_dir=(L3[0], L3[1])), seed=T['seed'],
                                  haze=1 - math.exp(-T['Z'] / 70.0), haze_col=(0.95, 0.74, 0.72))
            # mist skirt: the base dissolves into the deck
            yy_ = np.arange(PH, dtype=np.float32)[:, None]
            skirt = K._ss(by - 0.5 * ppu, by + 0.1 * ppu, yy_) * np.ones((1, PW), np.float32)
            nz_ = K._noise(PW, PH, max(PW / (0.8 * ppu), 3), T['seed'] + 5, 3, stretch=4.0)
            P[..., 3] *= 1 - np.clip(skirt * (0.75 + 0.35 * nz_), 0, 1)
            self.towers.append((P, T['Z']))
        # altocumulus streets in the upper sky, lit gold on the sun-facing (lower) edges
        self.flecks = K.sky_rows_plate(PW, PH, hy, seed=21, fov=FOV, alt=4.0, cell=0.34, cell_stretch=1.3,
                                       row_l=0.9, patch=6.0, patch_thr=0.12, cover=0.55, warp=1.6, angle=18.0,
                                       region=(0.0, 0.95), z_haze=50.0, lit=(1.3, 0.82, 0.5),
                                       body=(0.46, 0.4, 0.68), under=(0.26, 0.25, 0.52), a_lo=0.1, a_hi=0.3,
                                       lumps=0.8, lit_screen=(0.0, 3.5), opacity=0.95, haze_col=(1.0, 0.72, 0.6),
                                       haze_fade=0.2, veil=0.35, edge_px=1.0)
        # horizon: thin white-hot line (strongest under the sun) + narrow warm bloom
        Y = np.arange(H, dtype=np.float32)[:, None]
        X = np.arange(W, dtype=np.float32)[None, :]
        d = (Y - self.hz * H) / H
        wide = np.exp(-(d / 0.018) ** 2) * (0.25 + 0.75 * np.exp(-((X - self.sun[0]) / (0.25 * W)) ** 2))
        line = np.exp(-(d / 0.0016) ** 2) * (0.3 + 0.7 * np.exp(-((X - self.sun[0]) / (0.3 * W)) ** 2))
        self.hglow = (wide[..., None] * np.array([1.0, 0.7, 0.42], np.float32) * 0.22 +
                      line[..., None] * np.array([1.0, 0.95, 0.86], np.float32) * 0.9).astype(np.float32)
        # airbrushed foreground streaks (yn_02)
        ys = np.arange(PH, dtype=np.float32)[:, None] / PH
        xs = np.arange(PW, dtype=np.float32)[None, :] / PW
        n2 = K._noise(PW, PH, 5.0, 78, 3, stretch=10.0, angle=-24)
        streak = np.clip(n2 * 1.3 - 0.15, 0, 1) * K._ss(0.5, 0.75, ys) * (1 - K._ss(0.9, 1.0, ys))
        streak = streak * np.exp(-((xs - sun_p[0] / PW) / 0.4) ** 2) * 0.1
        self.streaks = np.dstack([np.broadcast_to(np.array([1.0, 0.72, 0.52], np.float32), (PH, PW, 3)),
                                  streak]).astype(np.float32)
        self.flare = K.SunFlare(W, H, tint=(1.0, 0.82, 0.58), star_len=0.07, bloom=0.55, ring=0.5, chain=1.2,
                                vertical=0.3)
        self._grid = np.mgrid[0:H, 0:W].astype(np.float32)
        self.setup_s = time.time() - t0

    def _warp_deck(self, cam, zoom):
        """Per-pixel parallax: plate -> screen with depth factor 1/Z (true multi-plane motion)."""
        W, H = self.W, self.H
        yy, xx = self._grid
        # first-order: sample depth at the mid-depth affine position
        d0 = 0.25
        z0 = 1 + (zoom - 1) * d0
        px = (xx - W / 2) / z0 + self.PW / 2 + cam[0] * d0
        py = (yy - H / 2) / z0 + self.PH / 2 + cam[1] * d0
        Zs = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dep = np.clip(2.2 / np.maximum(Zs, 0.5), 0.02, 1.5)
        z = 1 + (zoom - 1) * dep
        px = (xx - W / 2) / z + self.PW / 2 + cam[0] * dep
        py = (yy - H / 2) / z + self.PH / 2 + cam[1] * dep
        rgba = cv2.remap(self.deck_rgba, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Zw = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rgba, Zw

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.06 * p
        cam = (0.05 * W * p, -0.008 * H * p)
        img = self.sky.copy()
        z0 = 1 + (zoom - 1) * 0.03
        lx = W / 2 + (self.sun[0] - W / 2) * z0 + cam[0] * 0.03
        ly = H / 2 + (self.sun[1] - H / 2) * z0 + cam[1] * 0.03
        img = img + S.sun(W, H, lx, ly, radius=0.006, color=(1.0, 0.9, 0.7), intensity=1.3, glow_size=0.1, core=2.5)
        Lk = K.drift(self.flecks, W, H, t, speed=0.002, cam=cam, zoom=zoom, depth=0.06)
        img = K.composite(img, Lk)
        img = img + self.hglow
        deck, Zw = self._warp_deck(cam, zoom)
        img = K.composite(img, deck)
        occ_t = []
        for i, (P, Zt) in enumerate(self.towers):
            dep = 2.2 / Zt
            Lt = K.drift(P, W, H, t, speed=0.0006 * (1 + i), cam=cam, zoom=zoom, depth=dep, billow=0.0006,
                         billow_scale=0.015, billow_rate=0.25, seed=6 + i)
            vis = K._ss(Zt - 0.6, Zt + 0.6, Zw)               # hidden behind nearer deck crests
            Lt = Lt.copy()
            Lt[..., 3] *= vis
            img = K.composite(img, Lt)
            occ_t.append(Lt)
        Ls = K.drift(self.streaks, W, H, t, speed=0.003, cam=cam, zoom=zoom, depth=1.0)
        img = img + Ls[..., :3] * Ls[..., 3:4]
        occ = K.occluder(*occ_t)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.08, length=0.9, tint=(1.0, 0.74, 0.5),
                                   radius=0.07, t=t)
        vis = F.sun_visibility(K.occluder(*occ_t), lx, ly, 0.008 * W)
        shim = 1.0 + 0.05 * math.sin(1.9 * t) + 0.02 * math.sin(4.3 * t + 0.6)
        # anamorphic streak through the sun
        img = img + self._streak(lx, ly) * (0.6 + 0.4 * vis) * shim
        img = img + self.flare.render(lx, ly, intensity=(0.45 + 0.55 * vis) * shim, t=t,
                                      center=(W / 2 + cam[0] * 0.3, H / 2))
        img = F.bloom_soft(img, threshold=1.05, knee=0.25, strength=0.18, halation=0.1)
        img = F.shoulder(img, 0.9, 0.05)
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.004, vig=0.25, ca=0.0008)

    def _streak(self, lx, ly):
        if not hasattr(self, '_stk'):
            W, H = self.W, self.H
            yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32)
            dx = np.abs(xx - W) / W
            dy = np.abs(yy - H) / H
            s = np.exp(-(dy / 0.0028) ** 2) * (np.exp(-dx / 0.35) * 0.35 + np.exp(-dx / 0.06) * 0.6)
            s = s + np.exp(-(dy / 0.012) ** 2) * np.exp(-dx / 0.2) * 0.08
            self._stk = (s[..., None] * np.array([1.0, 0.8, 0.6], np.float32)).astype(np.float32)
        M = np.array([[1, 0, lx - self.W], [0, 1, ly - self.H]], np.float32)
        return cv2.warpAffine(self._stk, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
