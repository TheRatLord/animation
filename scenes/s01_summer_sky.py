"""s01_summer_sky - OPENING. A colossal summer cumulonimbus in a deep blue sky; the sun hides just
behind its crown, throwing light shafts and a lens flare; a jet contrail is slowly drawn across; birds
cross in front of the tower. The shot opens on backlit rooftops, antennas, a utility pole and sagging
wires in the bottom of frame and slowly tilts UP to the towering cloud (multi-plane parallax + a slight
push-in)."""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, sky as S, fx as F, clouds2 as K  # noqa: E402
import s01_summer_sky_hero as HR  # noqa: E402
import s01_summer_sky_vol as CB  # noqa: E402  (hero tower: clouds3 volumetric lobes + s01 paint)
import s01_summer_sky_city as T  # noqa: E402
import s01_summer_sky_fx as X  # noqa: E402
import s01_summer_sky_flare as FL  # noqa: E402

DURATION = 5.5

SKY_PRESET = dict(stops=[(0.0, '#0a45b8'), (0.16, '#155dcd'), (0.34, '#2e80de'), (0.5, '#58a2e9'),
                         (0.64, '#8ec4f0'), (0.78, '#b8dcf6'), (0.9, '#d6ecf9'), (1.0, '#ebf8fc')],
                  sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc', band=('#f4fbfd', 0.85))


class Plate:
    """A plate sampled by the tilting camera. depth scales the tilt travel (parallax)."""

    def __init__(self, img, depth, W, H, travel, mg):
        self.img = np.ascontiguousarray(img, np.float32)
        self.depth = depth
        self.hp, self.wp = img.shape[:2]
        self.W, self.H = W, H
        self.travel = travel
        self.mg = mg
        a = self.img[..., 3].max(1) if self.img.shape[2] == 4 else np.ones(self.hp)
        nz = np.nonzero(a > 1e-3)[0]
        self._rows = (nz.min(), nz.max()) if len(nz) else (0, -1)

    @staticmethod
    def size(W, H, depth, travel, mg, wfac=1.12):
        return int(W * wfac), int(H + depth * travel + 2 * mg)

    def center(self, u, dx=0.0):
        cy = self.hp - self.mg - self.H / 2 - self.depth * self.travel * u
        return self.wp / 2 + dx, cy

    def zoom(self, zoom):
        return 1 + (zoom - 1) * self.depth

    def to_screen(self, p, u, zoom, dx=0.0):
        z = self.zoom(zoom)
        cx, cy = self.center(u, dx)
        return (np.asarray(p[0]) - cx) * z + self.W / 2, (np.asarray(p[1]) - cy) * z + self.H / 2

    def to_plate(self, xs, ys, u, zoom, dx=0.0):
        z = self.zoom(zoom)
        cx, cy = self.center(u, dx)
        return (xs - self.W / 2) / z + cx, (ys - self.H / 2) / z + cy

    def on_screen(self, u, zoom):
        y0 = self.to_screen((0, self._rows[0]), u, zoom)[1]
        y1 = self.to_screen((0, self._rows[1]), u, zoom)[1]
        return y1 >= -2 and y0 <= self.H + 2

    def matrix(self, u, zoom, dx=0.0):
        z = self.zoom(zoom)
        cx, cy = self.center(u, dx)
        return np.float32([[z, 0, self.W / 2 - cx * z], [0, z, self.H / 2 - cy * z]])

    def render(self, u, zoom, dx=0.0, img=None):
        img = self.img if img is None else img
        bv = (0,) * img.shape[2]
        return cv2.warpAffine(img, self.matrix(u, zoom, dx), (self.W, self.H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=bv)


def _vnoise1(n, seed, smooth=3.0):
    rng = np.random.default_rng(seed)
    v = cv2.GaussianBlur(rng.standard_normal((1, n)).astype(np.float32), (0, 0), smooth)[0]
    return v / (v.std() + 1e-6)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.TR = 0.2 * H                  # foreground crane travel (px): slow, stately establishing move
        self.PT = 0.46 * H                  # pole placement (pole top at ~0.12 H in the opening frame)
        self.mg = 0.03 * H
        self.D_SKY = 0.8                  # sky + tower barely move: a crane-down reveal, not a pan
        self.D_POLE = 1.45                 # near pole + wires sweep ~1.75x faster than the rooftops (parallax)
        self._build_sky(W, H)
        self._build_fg(W, H)
        xs, ys = C.grid(W, H)
        self.xs, self.ys = xs, ys
        self.paper = self._paper(W, H)

    # ------------------------------------------------------------------ sky + tower
    def _build_sky(self, W, H):
        TR, mg, D = self.TR, self.mg, self.D_SKY
        pw, ph = Plate.size(W, H, D, TR, mg)
        self.pw, self.ph = pw, ph
        ox = (pw - W) / 2
        y_top0 = ph - mg - H                         # plate row at screen top at t=0
        y_top1 = mg                                  # ... at the end of the tilt
        base_y = y_top0 + 0.92 * H                   # cloud foot (far below the horizon: hidden by the town)
        self.hz_y = y_top0 + 0.8 * H                 # horizon
        top_y = y_top1 + 0.1 * H                   # crown top at the end (headroom for the sun + rays)
        self.base_y, self.top_y = base_y, top_y
        x0 = ox + 0.38 * W                            # left of centre: cloud + pole form a thirds layout
        Hc = (base_y - top_y) / 1.01
        self.Hc = Hc
        rgba, self.sun_p = CB.hero(pw, ph, x0, base_y, Hc, W, self.hz_y)
        self.tower = Plate(rgba, D, W, H, TR, mg)
        self.tower_axis = float(x0)
        # sky
        hzp = self.hz_y / ph
        sky = S.sky_gradient(pw, ph, SKY_PRESET, horizon=hzp, sun=self.sun_p, sun_radius=0.5)
        self.sky = Plate(sky, D, W, H, TR, mg)
        self._sky_rgb = sky
        # distant cumulus bank on the horizon (behind everything, hazy -> scale)
        bank = HR.bank(pw, ph, self.hz_y + 0.004 * H, W, self.sun_p)
        self.bank = Plate(bank, D, W, H, TR, mg)
        # soft fibrous cirrus streams in the open sky (x0, y0, length, angle, bend, half-width, opacity)
        streams = [(ox + 0.62 * W, y_top1 + 0.13 * H, 0.5 * W, -9, 0.04, 0.02 * H, 0.55),
                   (ox + 0.72 * W, y_top1 + 0.03 * H, 0.45 * W, -6, -0.03, 0.014 * H, 0.4),
                   (ox - 0.12 * W, y_top1 + 0.12 * H, 0.42 * W, -14, 0.05, 0.022 * H, 0.45),
                   (ox - 0.08 * W, y_top1 + 0.34 * H, 0.3 * W, -10, 0.03, 0.012 * H, 0.3),
                   (ox + 0.66 * W, y_top0 + 0.05 * H, 0.5 * W, -7, 0.03, 0.018 * H, 0.35),
                   (ox + 0.78 * W, y_top0 + 0.26 * H, 0.35 * W, -4, 0.02, 0.01 * H, 0.25),
                   (ox - 0.1 * W, y_top0 + 0.02 * H, 0.3 * W, -12, 0.04, 0.012 * H, 0.25)]
        cir = X.cirrus_soft(pw, ph, W, streams, seed=6, sun=self.sun_p)
        self.cirrus = Plate(cir, D * 0.97, W, H, TR, mg)
        # contrail path (plate coords): plane flies from the right edge up-left toward the crown
        self.tr_a = np.array([ox + 1.2 * W, y_top1 + 0.44 * H])
        self.tr_b = np.array([ox + 0.62 * W, y_top1 + 0.09 * H])
        self.tr_noise = _vnoise1(2048, 5, 80.0)
        self.tr_noise2 = _vnoise1(2048, 9, 60.0)
        self.bird_col = np.array([0.16, 0.2, 0.36], np.float32)

    # ------------------------------------------------------------------ foreground
    def _build_fg(self, W, H):
        TR, mg = self.TR, self.mg
        fw, fh = Plate.size(W, H, 0.9, TR, mg)
        T.GLINTS.clear()
        far, _ = T.town_far(W, H, fw, fh, fh - mg - H)
        self.far_glints = list(T.GLINTS)
        T.GLINTS.clear()
        self.town_far = Plate(self._full(far, fh, fh - mg - H + 0.5 * H), 0.9, W, H, TR, mg)
        nw, nh = Plate.size(W, H, 1.0, TR, mg)
        near, _ = T.town_near(W, H, nw, nh, nh - mg - H)
        self.near_glints = list(T.GLINTS)
        T.GLINTS.clear()
        self.town_near = Plate(self._full(near, nh, nh - mg - H + 0.5 * H), 1.0, W, H, TR, mg)
        qw, qh = Plate.size(W, H, self.D_POLE, TR, mg)
        ox = (qw - W) / 2
        pole, wires = T.pole_and_wires(W, H, qw, qh, qh - mg - H, self.PT, sun_x=ox + 0.47 * W)
        self.pole = Plate(pole, self.D_POLE, W, H, TR, mg)
        # specular glint points (pole-plate coords): where each wire crosses the sun's azimuth band, the
        # insulator tops and the transformer lid
        # specular glints: only where the top wires cross the sun's azimuth (2-3 small points)
        sun_x = ox + 0.47 * W
        gp = []
        for i, (pts, wd) in enumerate(wires):
            if len(pts) < 20 or len(gp) >= 3 or i % 2:
                continue
            k = int(np.argmin(np.abs(pts[:, 0] - (sun_x + (0.012 * i) * W))))
            if 2 < k < len(pts) - 3:
                gp.append((pts[k, 0], pts[k, 1] - wd * 0.4, 1.3 * i, 0.9, 0.55))
        self.glint_pts = gp
        # horizon haze band (pale cyan/white) between the far and near roofs and the sky, fixed to the
        # far-town depth
        hh = np.zeros((fh, fw, 4), np.float32)
        yb = fh - mg - H + 0.8 * H
        yy = np.arange(fh, dtype=np.float32)[:, None]
        a = np.exp(-((yy - yb) / (0.035 * H)) ** 2) * 0.55 + C.smoothstep(yb - 0.02 * H, yb + 0.08 * H, yy) * 0.25
        hh[..., :3] = np.array([0.88, 0.96, 1.0], np.float32)
        hh[..., 3] = np.broadcast_to(a, (fh, fw))
        self.haze_band = Plate(hh, 0.9, W, H, TR, mg)

    @staticmethod
    def _paper(W, H, seed=77):
        """Static paint/paper texture (a multiplier ~1 +- 1.5 %): broad uneven wash + fine fibre tooth.
        Fixed to the frame (like the grain of a painted cel under the camera), identical every frame."""
        rng = np.random.default_rng(seed)
        s = W / 1920.0
        broad = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 60 * s)
        broad /= broad.std() + 1e-6
        fib = rng.standard_normal((H, W)).astype(np.float32)
        fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=2.2 * s + 0.3, sigmaY=0.7 * s + 0.3)
        fib /= fib.std() + 1e-6
        tooth = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.8 * s + 0.2)
        tooth /= tooth.std() + 1e-6
        return (1.0 + 0.007 * broad + 0.0035 * fib + 0.003 * tooth).astype(np.float32)[..., None]

    @staticmethod
    def _full(part, full_h, y0):
        y0 = int(y0)
        out = np.zeros((full_h, part.shape[1], 4), np.float32)
        out[y0:y0 + part.shape[0]] = part[:full_h - y0]
        return out

    T0 = 0.5          # the camera holds on the rooftops for ~0.5 s before the tilt begins

    def cam(self, t):
        # a clear ease-in / ease-out crane (smootherstep: zero velocity AND acceleration at both ends),
        # settling on the tower just before the cut
        s = min(max((t - self.T0) / (DURATION - 0.25 - self.T0), 0.0), 1.0)
        u = s * s * s * (s * (6 * s - 15) + 10)
        zoom = 1.0 + 0.03 * u
        return u, zoom

    # ------------------------------------------------------------------ per-frame pieces
    def _tower(self, u, zoom, t, dx):
        """Tower plate with slow billowing: it grows upward (anchored at the base) and swells a little
        wider over the shot - a single smooth affine, so no warping artifacts."""
        P = self.tower
        cx, cy = P.center(u, dx)
        return S.drift(P.img, self.W, self.H, t, speed=0.0, zoom=zoom, depth=P.depth,
                       offset=(cx - P.wp / 2, cy - P.hp / 2), billow=0.0014, billow_scale=0.03,
                       billow_rate=0.28, seed=3)

    def _contrail(self, img, u, zoom, t, dx):
        """Jet contrail drawn analytically: sharp twin lines at the plane, widening, softening and
        breaking into puffs with age. The head advances visibly during the shot."""
        W, H = self.W, self.H
        P = self.sky
        a = np.array(P.to_screen(self.tr_a, u, zoom, dx), np.float64)
        b = np.array(P.to_screen(self.tr_b, u, zoom, dx), np.float64)
        v = b - a
        Lp = float(np.hypot(*v))
        d = v / Lp
        nrm = np.array([-d[1], d[0]])
        f_head = 0.12 + 0.8 * (t / DURATION)
        head = a + v * f_head
        x0 = int(max(min(a[0], head[0]) - 0.05 * W, 0))
        x1 = int(min(max(a[0], head[0]) + 0.05 * W, W))
        y0 = int(max(min(a[1], head[1]) - 0.05 * W, 0))
        y1 = int(min(max(a[1], head[1]) + 0.05 * W, H))
        if x1 <= x0 or y1 <= y0:
            return img, None
        xs = self.xs[y0:y1, x0:x1] - a[0]
        ys = self.ys[y0:y1, x0:x1] - a[1]
        s = (xs * d[0] + ys * d[1]) / Lp                # 0..1 along the path
        q = xs * nrm[0] + ys * nrm[1]                   # px across
        age = np.clip(f_head - s, 0, None)              # path fraction behind the plane
        idx = np.clip((s * 2047).astype(np.int32), 0, 2047)
        # the trail drifts (wind) and waves a little with age; it widens, softens and breaks into puffs
        q = q - age * 0.005 * W * self.tr_noise[idx] - (age ** 1.3) * 0.022 * W
        w0 = 0.0007 * W
        wd = w0 + age * 0.014 * W
        sep = 0.0015 * W * np.exp(-age / 0.2)
        prof = np.exp(-((q - sep) / wd) ** 2) + np.exp(-((q + sep) / wd) ** 2)
        prof = np.clip(prof, 0, 1.3)
        brk = np.clip(age * 3.5, 0, 1)
        puff = np.clip(1 + 0.55 * self.tr_noise2[idx] * brk, 0.15, 2.0)
        op = 0.95 * np.exp(-age / 0.75) * puff * C.smoothstep(0.0, 0.004, f_head - s + 0.0005) *             C.smoothstep(-0.001, 0.02, s + 0.02)
        head_fade = C.smoothstep(-0.003, 0.004, f_head - s)
        al = np.clip(prof * op * head_fade / (1 + age * 1.6), 0, 1)[..., None]
        col = np.array([1.08, 1.06, 1.02], np.float32)
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - al) + col * al
        return img, head

    BIRDS = [(-0.02, 0.0, 1.0, 0.0, 3.0), (0.05, -0.028, 0.82, 1.9, 3.5), (-0.1, 0.035, 0.78, 4.1, 2.8),
             (0.11, 0.022, 0.64, 2.7, 3.7), (-0.16, -0.035, 0.6, 5.3, 3.2), (0.17, -0.05, 0.5, 0.9, 3.9),
             (-0.06, -0.07, 0.55, 3.3, 3.3)]

    def _birds(self, img, u, zoom, t, dx):
        """Five gull-like silhouettes crossing left -> right in front of the tower, each with its own
        wingbeat (phase/rate) and a slow glide modulation; drawn supersampled in small boxes."""
        W, H = self.W, self.H
        # loose formation in the open sky right of the anvil tip, gliding right and climbing slowly
        cxp = self.pw / 2 + 0.24 * W + 0.02 * W * t
        cyp = self.hz_y - 0.8 * H - 0.02 * H * t
        for i, (ox_, oy_, sc, ph, fr) in enumerate(self.BIRDS):
            wob = math.sin(t * 0.9 + ph) * 0.004 * W
            p = self.sky.to_screen((cxp + ox_ * W, cyp + oy_ * W + wob), u, zoom, dx)
            bx, by = float(p[0]), float(p[1])
            span = 0.03 * W * (0.82 + 0.18 * sc)
            if bx < -span or bx > W + span or by < -span or by > H + span:
                continue
            glide = C.smoothstep(0.2, 0.7, 0.5 + 0.5 * math.sin(t * 0.7 + ph * 1.3))
            f = math.sin(t * fr * 2 * math.pi + ph) * (0.25 + 0.75 * glide)
            tip = -0.3 * f + 0.04
            elb = -0.1 - 0.12 * f
            wr = (elb + tip) * 0.5 + 0.02
            up = [(0.03, -0.035), (0.17, elb - 0.03), (0.33, wr - 0.02), (0.5, tip)]
            lo = [(0.5, tip + 0.012), (0.34, wr + 0.05), (0.18, elb + 0.07), (0.05, 0.05)]
            wing = np.array(up + lo, np.float64)
            wing[:, 0] = wing[:, 0] - 0.06 * np.array([0, 0.3, 0.7, 1.0, 1.0, 0.7, 0.3, 0]) * 0
            ss_ = 4
            R_ = int(span * 0.7) + 3
            x0, y0 = int(bx) - R_, int(by) - R_
            x1, y1 = int(bx) + R_ + 1, int(by) + R_ + 1
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            m = np.zeros(((y1 - y0) * ss_, (x1 - x0) * ss_), np.uint8)
            org = np.array([bx - x0, by - y0])
            for sgn in (1, -1):
                pts = wing * [sgn * span, span] + org
                cv2.fillPoly(m, [(pts * ss_ * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            body = np.array([(0.16, -0.005), (0.1, -0.03), (-0.06, -0.025), (-0.2, 0.0), (-0.24, 0.035),
                             (-0.15, 0.03), (0.06, 0.03), (0.13, 0.02)]) * [span * 0.55, span] + org
            cv2.fillPoly(m, [(body * ss_ * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            a = cv2.resize(m.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            a = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            r = (slice(cy0, cy1), slice(cx0, cx1))
            # atmospheric tint: a deep blue-grey pulled toward the local sky (never flat black)
            bg = img[r].reshape(-1, 3).mean(0)
            bc = self.bird_col * 0.62 + bg * 0.38 * (0.75 + 0.25 * (1 - sc))
            img[r] = img[r] * (1 - a[..., None]) + bc * a[..., None]
        return img

    RAYS = [(-172, 3.0, 0.7, 0.5), (-158, 2.0, 0.5, 0.3), (-140, 4.5, 0.8, 0.7), (-118, 2.5, 0.45, 0.25),
            (-62, 3.0, 0.5, 0.35), (-40, 5.0, 0.85, 0.8), (-22, 2.2, 0.6, 0.3), (-8, 3.5, 0.9, 0.6),
            (8, 2.5, 0.7, 0.4), (21, 4.0, 1.0, 0.75), (34, 2.0, 0.55, 0.22), (48, 3.5, 0.75, 0.5),
            (140, 3.0, 0.8, 0.45), (158, 4.5, 1.0, 0.65), (171, 2.2, 0.6, 0.28)]

    def _rays(self, w, h, lx, ly, occ, t):
        """Crepuscular rays: soft additive wedges fanning from the hidden sun into the blue, occluded
        by the cloud (radially smeared occlusion so a ray is cut where the cloud blocks it)."""
        q = 2
        ww, hh = w // q, h // q
        xs, ys = C.grid(ww, hh)
        dx, dy = xs - lx / q, ys - ly / q
        r = np.sqrt(dx * dx + dy * dy) / ww
        ang = np.degrees(np.arctan2(dy, dx))
        pat = np.zeros_like(r)
        rot = 0.25 * t
        for a0, wd, k, ln in self.RAYS:
            da = (ang - a0 - rot + 180) % 360 - 180
            wdr = wd * (1 + 0.6 * np.clip(r / 0.5, 0, 1))
            pat += k * np.exp(-(da / wdr) ** 2) * np.exp(-r / ln)
        pat *= C.smoothstep(0.01, 0.06, r) * (0.95 + 0.05 * math.sin(t * 0.6))
        o = cv2.resize(occ, (ww, hh), interpolation=cv2.INTER_AREA)
        pat = pat * (1 - np.clip(C.blur(o, 0.01 * ww) * 1.3, 0, 1))
        out = cv2.resize(C.blur(pat.astype(np.float32), 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        return out[..., None] * np.array([1.0, 0.96, 0.86], np.float32) * 0.3

    def _glint(self, img, x, y, L, inten, color=(1.0, 0.95, 0.82)):
        W, H = self.W, self.H
        R = int(L) + 2
        if x < -R or y < -R or x > W + R or y > H + R:
            return
        x0, y0 = int(x) - R, int(y) - R
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        g = F.glints(2 * R + 1, 2 * R + 1, [x - x0], [y - y0], size=L / (2 * R + 1), intensity=inten, color=color,
                     angle=0.12)
        img[cy0:cy1, cx0:cx1] += g[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]

    def _qgrid(self, q):
        if not hasattr(self, '_qg'):
            self._qg = {}
        if q not in self._qg:
            xs, ys = C.grid(self.W // q, self.H // q)
            self._qg[q] = ((xs + 0.5) * q - 0.5, (ys + 0.5) * q - 0.5)
        return self._qg[q]

    def frame(self, t):
        W, H = self.W, self.H
        u, zoom = self.cam(t)
        drift = 0.008 * W * t / DURATION
        img = self.sky.render(u, zoom)[..., :3].copy()
        lx, ly = self.sky.to_screen(self.sun_p, u, zoom)
        lx, ly = float(lx), float(ly)
        img = F.over_rgba(img, self.cirrus.render(u, zoom, dx=0.4 * drift))
        img = F.over_rgba(img, self.bank.render(u, zoom, dx=0.3 * drift))
        img, head = self._contrail(img, u, zoom, t, 0.0)
        tw = self._tower(u, zoom, t, drift)
        occ = tw[..., 3]
        self._sun_vis = 0.3 + 0.7 * float(F.sun_visibility(occ, lx, ly, 0.012 * W))
        img = F.over_rgba(img, tw)
        img = self._birds(img, u, zoom, t, drift * 0.8)
        # foreground (skipped once it has left the frame)
        for P in (self.haze_band, self.town_far, self.town_near, self.pole):
            if P.on_screen(u, zoom):
                img = F.over_rgba(img, P.render(u, zoom))
        # twinkling specular glints on wires / insulators / transformer lid (small boxes)
        if self.pole.on_screen(u, zoom):
            for (gx_, gy_, ph_, fr_, amp_) in self.glint_pts:
                sx_, sy_ = self.pole.to_screen((gx_, gy_), u, zoom)
                tw_ = amp_ * (0.7 + 0.3 * math.sin(t * fr_ + ph_))
                self._glint(img, float(sx_), float(sy_), 0.006 * W, tw_)
        # plane glint at the contrail head (small box only)
        if head is not None:
            gx0, gy0 = int(head[0]) - 24, int(head[1]) - 24
            if 0 <= gx0 and 0 <= gy0 and gx0 + 49 <= W and gy0 + 49 <= H:
                g = F.glints(49, 49, [head[0] - gx0], [head[1] - gy0], size=0.014 * W / 49,
                             intensity=1.5 + 0.15 * math.sin(t * 2.0), color=(1.0, 0.97, 0.9))
                img[gy0:gy0 + 49, gx0:gx0 + 49] += g
                C.splat(img, [head[0]], [head[1]], 0.0022 * W, (1.0, 0.98, 0.92), 2.0)
        # ---- all soft light at half resolution: sun glow spilling over the crown, crepuscular rays
        # fanning out from behind the cloud edge (occlusion-driven), highlight bloom, lens flare core +
        # a chain of hexagonal ghosts that slides along the sun -> centre axis as the camera tilts
        w2, h2 = W // 2, H // 2
        sun_in = float(C.smoothstep(-0.12 * H, 0.08 * H, ly))           # ghosts only once the sun enters frame
        sm = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        occh = cv2.resize(occ, (w2, h2), interpolation=cv2.INTER_AREA)
        acc = F.bloom_soft(sm, threshold=1.3, knee=0.15, strength=0.15, halation=0.03) - sm
        xq, yq = self._qgrid(2)
        d = np.sqrt((xq - lx) ** 2 + (yq - ly) ** 2) / W
        glow = (np.exp(-d / 0.008) * 0.2 + np.exp(-d / 0.1) * 0.045 + 1.0 / (1 + (d / 0.003) ** 2) * 0.5)
        sky_l = glow[..., None] * np.array([1.0, 0.96, 0.86], np.float32)
        w4, h4 = W // 4, H // 4
        occq = cv2.resize(occh, (w4, h4), interpolation=cv2.INTER_AREA)
        sky_l += cv2.resize(X.crepuscular(w4, h4, lx / 4, ly / 4, occq, strength=0.3, length=0.95, radius=0.045, t=t,
                                                  tint=(1.0, 0.86, 0.62)),
                            (w2, h2), interpolation=cv2.INTER_LINEAR)
        sky_l += self._rays(w2, h2, lx / 2, ly / 2, occh, t) * 0.35
        occs = np.clip(occh * 1.15, 0, 1)
        # sky-only light is masked at FULL resolution below, so the cloud's silhouette (and its thin
        # lining) stays crisp instead of being fogged by a half-res glow fringe
        acc_sky = sky_l
        # a narrow glare only right at the rim nearest the sun
        acc += (1.0 / (1 + (d / 0.01) ** 2) * 0.12 * occs)[..., None] * np.array([1.0, 0.95, 0.85], np.float32)
        # soft halation where the sun meets the cloud top: warm glow bleeding over the crown's paint
        acc += ((np.exp(-d / 0.025) * 0.1 + np.exp(-d / 0.08) * 0.02) * occs)[..., None] *             np.array([1.0, 0.9, 0.74], np.float32)
        acc_fl = F.anime_flare(w2, h2, lx / 2, ly / 2, intensity=0.42 * (0.4 + 0.6 * sun_in), tint=(1.0, 0.94, 0.84), rays=6, ray_len=0.05,
                             starburst=0.45, ghosts=0.0, halo=0.0, streak=1.0, glow=0.15, rot=0.01 * t)
        if sun_in > 0.01:
            # ghost chain along the sun -> optical-centre axis (slides as the camera tilts) + anamorphic streak
            acc_fl = acc_fl + FL.ghost_chain(w2, h2, lx / 2, ly / 2, (0.66 * W + 0.03 * W * (u - 0.5)) / 2, 0.52 * H / 2,
                                  amt=0.7 * sun_in * (0.55 + 0.45 * self._sun_vis))
            acc += FL.anamorphic(w2, h2, lx / 2, ly / 2, amt=0.6 * sun_in * (0.4 + 0.6 * self._sun_vis), xs=xq / 2, ys=yq / 2)
        img = img + cv2.resize(acc.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        occf = np.clip(occ * 1.1, 0, 1)[..., None]
        img = img + cv2.resize(acc_sky.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * (1 - occf) ** 2
        img = img + cv2.resize(acc_fl.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * (1 - 0.85 * occf)
        img = F.shoulder(img, 0.96, desat=0.04)
        # paper tooth: subtle everywhere, halved again on the cloud paint
        img = img * (1 + (self.paper - 1) * (1 - 0.6 * occ[..., None]))
        return F.finish_fast(img, t, sat=1.07, grain_amt=0.0, vig=0.12, ca=0.0003)
