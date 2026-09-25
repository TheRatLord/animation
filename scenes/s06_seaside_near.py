"""Near-foreground plates for s06_seaside (the camera trucks past them: strongest parallax).

  back : dark leaf mass + weeds in the bottom-left corner, a concrete utility pole a few metres from the
         lens (street-lamp arm, drop cable, step bolts, a pole advert and an address plate with real
         Japanese text), a galvanised pipe fence along the bottom right;
  front: backlit susuki (Japanese pampas grass) plumes and long blades rising into the frame (sway).
Canvases are W + 2 * margin wide (frame x at mid-shot + margin), H tall; each element carries its
inverse depth (Canvas zmode) for the truck warp."""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from lib import core as C
import s06_seaside_paint as P
import s06_seaside_foliage as FO
import s06_seaside_leaf as LF
import s06_seaside_tree as TR
import s06_seaside_tree9 as T9

FONT = 'C:/Windows/Fonts/YuGothB.ttc'
Z_POLE = 5.4
Z_GRASS = 3.9
Z_BUSH = 9.0
TOP = 0.8       # canvas extension above the frame (fraction of H): revealed by the crane-down


def cc(h):
    return C.hex2rgb(h)


def text_mask(text, size_px, vertical=True, font=FONT, spacing=1.05):
    """Anti-aliased text coverage (float32 0..1) rendered with a real Japanese font."""
    size_px = max(int(round(size_px)), 6)
    ss = 3
    fnt = ImageFont.truetype(font, size_px * ss, index=0)
    if vertical:
        cw = int(size_px * ss * 1.1)
        step = int(size_px * ss * spacing)
        img = Image.new('L', (cw, step * len(text) + size_px * ss // 4), 0)
        d = ImageDraw.Draw(img)
        for i, ch in enumerate(text):
            bb = d.textbbox((0, 0), ch, font=fnt)
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            x = (cw - w) / 2 - bb[0]
            y = i * step + (step - h) / 2 - bb[1]
            d.text((x, y), ch, font=fnt, fill=255)
    else:
        d0 = ImageDraw.Draw(Image.new('L', (4, 4)))
        bb = d0.textbbox((0, 0), text, font=fnt)
        img = Image.new('L', (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
        d = ImageDraw.Draw(img)
        d.text((4 - bb[0], 4 - bb[1]), text, font=fnt, fill=255)
    a = np.asarray(img, np.float32) / 255.0
    h, w = a.shape
    return cv2.resize(a, (max(w // ss, 1), max(h // ss, 1)), interpolation=cv2.INTER_AREA)


def stamp(cv, mask, x, y, color, alpha=1.0, anchor='c'):
    """composite a coverage mask at (x, y) (anchor c = centre, t = top-centre)."""
    h, w = mask.shape
    x0 = int(round(x - w / 2))
    y0 = int(round(y - (h / 2 if anchor == 'c' else 0)))
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, cv.W), min(y0 + h, cv.H)
    if X1 <= X0 or Y1 <= Y0:
        return
    m = mask[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    cv.put((m, X0, Y0), color, alpha)


class Near:
    def __init__(self, sc):
        self.sc = sc
        W, H = sc.W, sc.H
        self.u = W / 1920.0
        self.mx = int(0.2 * W)
        self.T = int(round(TOP * H))
        self.cw, self.ch = W + 2 * self.mx, H + self.T
        self.hz = sc.HZ * H + self.T    # horizon row in canvas px (canvas row = frame row + T)
        self.f = sc.f
        self.hc = sc.hc

    def X(self, fx):
        return self.mx + fx * self.sc.W

    def yof(self, Y, Z):
        return self.hz + self.f * (self.hc - Y) / Z

    def build(self):
        back = P.Canvas(self.cw, self.ch)
        front = P.Canvas(self.cw, self.ch)
        wgt = P.Canvas(self.cw, self.ch)
        self._bush(back)
        self._pole(back)
        # (the foreground pipe fence is left out: at the end of the crane it entered as two flat bars)
        self._pampas(front, wgt)
        w = wgt.straight()
        return dict(back=back.straight(), front=front.straight(), front_w=w[..., 0] * w[..., 3],
                    iz_back=back.iz_field(25.0, fill=1.0 / Z_POLE), iz_front=1.0 / Z_GRASS)

    # ------------------------------------------------------------------ bottom-left bush + weeds
    def _bush(self, cv):
        W, H = self.sc.W, self.sc.H
        u = self.u
        rng = np.random.default_rng(401)
        cv.zmode = ('c', 1.0 / Z_BUSH)
        pal = dict(deep=cc('#070d15'), shd=cc('#0e1824'), sky=cc('#1d3040'), lit=cc('#5a4632'),
                   hot=cc('#c47a4c'), rim=np.array([1.3, 0.72, 0.4], np.float32))
        # one backlit leaf mass (s06_seaside_tree): cool dark body, warm crowns toward the sun only
        T = self.T
        lob = [(self.X(-0.06), 0.98 * H + T, 0.1 * W, 0.0, 0), (self.X(0.02), 0.93 * H + T, 0.075 * W, 0.0, 0),
               (self.X(0.075), 0.99 * H + T, 0.06 * W, 0.0, 0), (self.X(0.12), 1.04 * H + T, 0.05 * W, 0.0, 0),
               (self.X(-0.02), 1.04 * H + T, 0.08 * W, 0.0, 0)]
        T9.canopy9(cv, lob, rng, unit=u * 2.0, far_haze=0.0, holes=0, expo=0.6, ex_floor=0.08)
        dark = cc('#101824')
        rim = cc('#ffab66')
        for k in range(30):
            P.grass_tuft(cv, self.X(rng.uniform(-0.06, 0.2)), rng.uniform(0.95, 1.06) * H + T,
                         H * rng.uniform(0.07, 0.18), rng, dark, rim, lean=0.2, n=8, rim_amt=0.7)

    # ------------------------------------------------------------------ utility pole
    def _pole(self, cv):
        sc = self.sc
        W, H = sc.W, sc.H
        u = self.u
        f, Z = self.f, Z_POLE
        cv.zmode = ('c', 1.0 / Z)
        xc = self.X(0.905)
        y_top, y_bot = -0.02 * H, 1.05 * H + self.T
        Ybot = self.hc - (y_bot - self.hz) * Z / f
        Ytop = self.hc - (y_top - self.hz) * Z / f
        rb, rt = 0.165, 0.165 - 0.0075 * (Ytop - Ybot) / 1.0 * 0.12     # gentle taper
        wb, wt = f * rb / Z, f * rt / Z
        pts = [(xc - wb, y_bot), (xc - wt, y_top), (xc + wt, y_top), (xc + wb, y_bot)]
        r = cv.poly_mask(pts, ss=4)
        m, x0, y0 = r
        h, w = m.shape
        gx, gy = cv.grid(x0, y0, w, h)
        vy = np.clip((y_bot - gy) / (y_bot - y_top), 0, 1)
        hw = wb + (wt - wb) * vy
        nn = np.clip((gx - xc) / hw, -1, 1)
        # backlit concrete: cool body, a warm rim on the sun side (left: the sun is left of the pole)
        body = C.lerp(cc('#2a2c48'), cc('#3a3656'), C.smoothstep(-0.9, 0.4, -nn)[..., None])
        body = C.lerp(body, cc('#1c1e34'), C.smoothstep(0.3, 1.0, nn)[..., None])
        # subtle vertical gradient: sky-lit and a touch warmer high up, darker toward the ground
        vg = C.smoothstep(0.0, 1.0, vy)[..., None]
        body = body * (0.86 + 0.2 * vg) + cc('#3a2c40') * (0.06 * (1 - vg))
        # concrete: broad mottling + fine pitting + a few vertical grime / rain streaks
        tex = C.fbm(w, h, 6, 4, seed=402)
        fine = C.fbm(w, h, max(int(w / (5.0 * u)), 4), 2, seed=405)
        body = body * (0.93 + 0.12 * tex[..., None]) * (0.95 + 0.1 * fine[..., None])
        rs = np.random.default_rng(406)
        streak = np.zeros((h, w), np.float32)
        for _ in range(9):
            sx_ = rs.uniform(0.08, 0.92) * w
            sy0 = rs.uniform(0.0, 0.7) * h
            ln = rs.uniform(0.08, 0.35) * h
            sw = rs.uniform(1.5, 5.0) * u
            fall = np.clip((gy - y0 - sy0) / ln, 0, 1)
            streak += np.exp(-((gx - x0 - sx_ - 3 * u * np.sin((gy - y0) / (40 * u))) / sw) ** 2) *                 (gy - y0 > sy0) * (1 - fall) ** 1.5 * rs.uniform(0.4, 1.0)
        body = body * (1 - 0.22 * np.clip(streak, 0, 1)[..., None])
        # a strong warm rim on the sun-facing (left) edge, with a wider warm falloff across the curve
        rimw = 5.5 * u / hw
        rim = np.clip(1 - (nn + 1) / (rimw * 2.0), 0, 1)
        glow = np.clip(1 - (nn + 1) / 0.45, 0, 1)
        body = body + cc('#ff9a66') * (glow ** 1.4 * 0.55)[..., None]
        body = body + (np.array([1.7, 0.95, 0.5], np.float32) - body) * (rim ** 1.2 * 0.95)[..., None]
        # cool sky bounce on the far edge
        body = body + cc('#5a5aa0') * (C.smoothstep(0.75, 1.0, nn) * 0.12)[..., None]
        # a crisp warm specular stripe just inside the rim (glazed concrete catching the low sun)
        spec = np.exp(-((nn + 0.7) / 0.07) ** 2) * (0.55 + 0.45 * vg[..., 0]) * (0.8 + 0.4 * tex)
        body = body + np.array([1.25, 0.66, 0.34], np.float32) * (0.42 * spec)[..., None]
        # faint horizontal casting joints (a lit notch on the sun side, dark line across)
        for fy_ in (0.18, 0.43, 0.71):
            jl = np.exp(-(((gy - y0) / h - fy_) * h / (1.1 * u)) ** 2)
            body = body * (1 - 0.3 * jl * C.smoothstep(-0.8, 0.2, nn))[..., None]
            body = body + np.array([1.0, 0.55, 0.3], np.float32) * (0.5 * jl * rim)[..., None]
        # lost, soft edge on the shade (right) side
        dR = (1 - nn) * hw
        r = (m * np.clip(dR / (3.2 * u) + 0.1, 0, 1) ** 0.8, x0, y0)
        cv.put(r, body)
        self.pole = (xc, wb, wt, y_top, y_bot)

        def px(Y):
            return self.yof(Y, Z)

        def half(yy):
            vv = np.clip((y_bot - yy) / (y_bot - y_top), 0, 1)
            return wb + (wt - wb) * vv
        # step bolts (alternating sides): they pierce the pole a little in front of its silhouette, with a
        # nut on the surface and a soft cast shadow below the bolt on the concrete
        for k in range(14):
            Y = Ybot + 0.35 + k * 0.45
            yy = px(Y)
            if yy < y_top or yy > y_bot:
                continue
            side = -1 if k % 2 == 0 else 1
            hwk = half(yy)
            L = f * 0.2 / Z
            th = max(f * 0.022 / Z, 1.0)
            xs0 = xc + side * hwk * 0.72
            # cast shadow on the pole (down and away from the sun)
            cv.line([(xs0 + side * th, yy + th * 1.3), (xc + side * hwk, yy + th * 1.7)], th * 1.1, cc('#12121e'), 0.55)
            cv.line([(xs0, yy), (xc + side * (hwk + L), yy)], th, cc('#26243c'))
            cv.line([(xc + side * (hwk + L), yy - th * 0.5), (xc + side * (hwk + L), yy - th * 1.6)], th * 0.8,
                    cc('#26243c'))
            # rust bleeding down the concrete from the bolt (tapered, broken streaks)
            rr_ = np.random.default_rng(900 + k)
            for j in range(int(rr_.integers(1, 3))):
                ox_ = xs0 + rr_.normal(0, th * 0.6)
                ln_ = f * rr_.uniform(0.12, 0.3) / Z
                pts_ = [(ox_ + 0.12 * th * math.sin(q * 0.9 + j), yy + th + q * ln_ / 4) for q in range(5)]
                for q in range(4):
                    cv.line(pts_[q:q + 2], th * (0.5 - 0.08 * q), cc('#5e3428'), 0.4 * (1 - q / 4.5))
            # nut where the bolt enters the concrete
            cv.poly([(xs0 - th * 0.9, yy - th * 1.1), (xs0 + th * 0.9, yy - th * 1.1), (xs0 + th * 0.9, yy + th * 1.1),
                     (xs0 - th * 0.9, yy + th * 1.1)], cc('#1c1a2c'))
            if side < 0:
                cv.line([(xc + side * (hwk * 0.9), yy - th * 0.45), (xc + side * (hwk + L), yy - th * 0.45)],
                        th * 0.35, cc('#ffb27a'), 0.9)
                cv.line([(xs0 - th * 0.9, yy - th * 1.1), (xs0 - th * 0.9, yy + th * 1.1)], th * 0.3, cc('#ff9e70'),
                        0.7)
        # small pole ID plate + two weathered stickers (in shade) - real Japanese text
        yy = px(6.3)
        hwk = half(yy)
        pw_, ph_ = hwk * 0.42, f * 0.2 / Z
        xp = xc - hwk * 0.28
        cv.poly([(xp - pw_, yy), (xp + pw_, yy), (xp + pw_, yy + ph_), (xp - pw_, yy + ph_)], cc('#aaa6b4') * 0.5)
        m = text_mask('汐見', min(ph_ * 0.3, pw_ * 0.8), vertical=False)
        stamp(cv, m, xp, yy + ph_ * 0.3, cc('#18203a'))
        m = text_mask('15', min(ph_ * 0.3, pw_ * 0.8), vertical=False)
        stamp(cv, m, xp, yy + ph_ * 0.72, cc('#18203a'))
        cv.line([(xp - pw_, yy), (xp - pw_, yy + ph_)], 1.3 * u, np.array([1.3, 0.75, 0.45], np.float32), 0.8)
        cv.line([(xp - pw_, yy), (xp + pw_, yy)], 1.0 * u, cc('#8a7a88'), 0.7)
        yy = px(5.9)
        hwk = half(yy)
        sh_ = f * 0.08 / Z
        cv.poly([(xc + hwk * 0.15, yy), (xc + hwk * 0.55, yy + 2 * u), (xc + hwk * 0.53, yy + sh_),
                 (xc + hwk * 0.13, yy + sh_ - 2 * u)], cc('#a88a48') * 0.4, 0.85)
        yy = px(6.7)
        cv.poly([(xc - hwk * 0.1, yy), (xc + hwk * 0.3, yy - 1 * u), (xc + hwk * 0.3, yy + sh_ * 0.8),
                 (xc - hwk * 0.1, yy + sh_ * 0.8)], cc('#a04a50') * 0.35, 0.7)
        # band clamps
        for Y in (5.35, 5.6):
            yy = px(Y)
            hwk = half(yy)
            cv.poly([(xc - hwk - 2 * u, yy - 5 * u), (xc + hwk + 2 * u, yy - 5 * u), (xc + hwk + 2 * u, yy + 5 * u),
                     (xc - hwk - 2 * u, yy + 5 * u)], cc('#34324c'))
            cv.line([(xc - hwk - 2 * u, yy - 5 * u), (xc - hwk * 0.2, yy - 5 * u)], 1.2 * u, cc('#ffb27a'), 0.8)
        # street-lamp arm: a real steel pipe (tapering 9 -> 6 cm) rising from a clamp collar and curving out to
        # the left, painted as a tube: sky-lit top half, dark core shadow low, a hot warm rim along the sun-side
        # top edge; a short straight brace under the bend; a flat LED housing with a lit top face, dark side,
        # glowing lens and end cap
        y0a = px(5.6)
        tt = np.linspace(0, 1, 60)
        axs = xc - half(y0a) * 0.6 - tt * f * 1.35 / Z
        ays = y0a - (1 - (1 - tt) ** 2.2) * f * 0.55 / Z
        thk = f * 0.045 / Z
        wa = f * (0.09 - 0.03 * tt) / Z                     # full width of the pipe along the arm
        dx_, dy_ = np.gradient(axs), np.gradient(ays)
        dl = np.hypot(dx_, dy_) + 1e-9
        nx_, ny_ = dy_ / dl, -dx_ / dl                      # normal pointing up (the arm runs leftward)
        if ny_.mean() > 0:
            nx_, ny_ = -nx_, -ny_

        def band(o0, o1, color, alpha=1.0):
            top = np.stack([axs + nx_ * wa * o1, ays + ny_ * wa * o1], 1)
            bot = np.stack([axs + nx_ * wa * o0, ays + ny_ * wa * o0], 1)
            cv.poly(np.concatenate([top, bot[::-1]], 0), color, alpha)
        # brace: a thinner straight strut from lower on the pole to the arm's bend
        yb_ = px(5.42)
        k_ = 11
        bx0, by0 = xc - half(yb_) * 0.8, yb_
        bw_ = f * 0.035 / Z
        cv.line([(bx0, by0), (axs[k_], ays[k_] + wa[k_] * 0.3)], bw_, cc('#24223a'))
        cv.line([(bx0, by0 - bw_ * 0.3), (axs[k_], ays[k_] + wa[k_] * 0.3 - bw_ * 0.3)], max(bw_ * 0.25, 1.0),
                cc('#ff9e70'), 0.6)
        band(-0.5, 0.5, cc('#262440'))                      # body (shade: backlit steel)
        band(0.05, 0.42, cc('#3c3a5c'), 0.9)                # sky-lit upper half
        band(-0.42, -0.18, cc('#1a1830'), 0.7)              # core shadow low on the tube
        band(0.3, 0.5, np.array([1.45, 0.86, 0.5], np.float32), 0.9)   # warm rim on the top (sun) edge
        band(0.42, 0.5, np.array([1.8, 1.25, 0.8], np.float32), 0.8)   # hot specular line
        band(-0.5, -0.4, cc('#5a4a70'), 0.35)               # faint bounce on the underside
        # joint collars (pipe reducer sleeves) along the arm: slightly proud of the tube, own rim + seams
        for k in (8, 30, 50):
            tx_, ty_ = dx_[k] / dl[k], dy_[k] / dl[k]
            cw2 = f * 0.028 / Z
            ww2 = wa[k] * 0.62
            px0, py0 = axs[k], ays[k]
            quad_ = [(px0 - tx_ * cw2 + nx_[k] * ww2, py0 - ty_ * cw2 + ny_[k] * ww2),
                     (px0 + tx_ * cw2 + nx_[k] * ww2, py0 + ty_ * cw2 + ny_[k] * ww2),
                     (px0 + tx_ * cw2 - nx_[k] * ww2, py0 + ty_ * cw2 - ny_[k] * ww2),
                     (px0 - tx_ * cw2 - nx_[k] * ww2, py0 - ty_ * cw2 - ny_[k] * ww2)]
            cv.poly(quad_, cc('#2c2a46'))
            cv.line([quad_[0], quad_[1]], 1.6 * u, np.array([1.7, 1.12, 0.7], np.float32), 0.9)
            cv.line([quad_[1], quad_[2]], 1.0 * u, cc('#16142a'), 0.8)
            cv.line([quad_[0], quad_[3]], 1.0 * u, cc('#6a5a80'), 0.5)
        # bolt plate where the brace meets the pole: a flat bracket with four bolt heads catching the light
        hwb = half(by0)
        pl_h = f * 0.07 / Z
        cv.poly([(xc - hwb - 5 * u, by0 - pl_h), (xc - hwb * 0.1, by0 - pl_h), (xc - hwb * 0.1, by0 + pl_h),
                 (xc - hwb - 5 * u, by0 + pl_h)], cc('#34304e'))
        cv.line([(xc - hwb - 5 * u, by0 - pl_h), (xc - hwb * 0.1, by0 - pl_h)], 1.4 * u, cc('#ffb27a'), 0.85)
        cv.line([(xc - hwb - 5 * u, by0 - pl_h), (xc - hwb - 5 * u, by0 + pl_h)], 1.4 * u, cc('#ff9e70'), 0.6)
        for (bxo, byo) in ((0.25, -0.55), (0.25, 0.55), (0.75, -0.55), (0.75, 0.55)):
            bx_b = xc - hwb - 5 * u + (hwb * 0.9 + 5 * u) * bxo
            by_b = by0 + pl_h * byo
            cv.line([(bx_b - 0.01, by_b), (bx_b + 0.01, by_b)], 4.2 * u, cc('#201e34'))
            cv.line([(bx_b - 0.8 * u, by_b - 0.9 * u), (bx_b + 0.4 * u, by_b - 1.1 * u)], 1.3 * u, cc('#ffc890'), 0.9)
        # clamp collar where the arm meets the pole
        yc_ = y0a
        hwk = half(yc_)
        cw_ = f * 0.055 / Z
        cv.poly([(xc - hwk - 3 * u, yc_ - cw_), (xc + hwk + 3 * u, yc_ - cw_), (xc + hwk + 3 * u, yc_ + cw_),
                 (xc - hwk - 3 * u, yc_ + cw_)], cc('#302e4a'))
        cv.line([(xc - hwk - 3 * u, yc_ - cw_), (xc - hwk - 3 * u, yc_ + cw_)], 1.8 * u, cc('#ffb07a'), 0.9)
        cv.line([(xc - hwk - 3 * u, yc_ - cw_), (xc + hwk * 0.2, yc_ - cw_)], 1.2 * u, cc('#ffb07a'), 0.7)
        # bolt heads on the arm clamp collar
        for k_ in range(3):
            yb2 = y0a + (k_ - 1) * f * 0.03 / Z
            xb2 = xc - hwk - 3 * u + 4.0 * u
            cv.line([(xb2 - 0.01, yb2), (xb2 + 0.01, yb2)], 3.6 * u, cc('#1c1a30'))
            cv.line([(xb2 - 0.9 * u, yb2 - 0.8 * u), (xb2 + 0.3 * u, yb2 - 1.0 * u)], 1.1 * u, cc('#ffc890'), 0.9)
        # LED housing (flat cobra head), slightly tilted with the arm tip
        hx, hy = axs[-1], ays[-1]
        hl = f * 0.5 / Z
        ht = f * 0.11 / Z
        sl_ = (ays[-1] - ays[-4]) / (axs[-1] - axs[-4] + 1e-9)
        def hp(xo, yo):
            return (hx - xo, hy + yo - xo * sl_ * 0.5)
        # socket sleeve over the pipe end
        cv.poly([hp(-hl * 0.02, -wa[-1] * 0.7), hp(hl * 0.14, -wa[-1] * 0.75), hp(hl * 0.14, wa[-1] * 0.75),
                 hp(-hl * 0.02, wa[-1] * 0.7)], cc('#2e2c48'))
        body_ = [hp(hl * 0.1, -ht * 0.55), hp(hl * 0.95, -ht * 0.35), hp(hl * 1.02, ht * 0.2), hp(hl * 0.96, ht * 0.5),
                 hp(hl * 0.1, ht * 0.5)]
        cv.poly(body_, cc('#24223c'))
        # lit top face (a thin wedge catching the sunset), then the rim line on its edge
        cv.poly([hp(hl * 0.1, -ht * 0.55), hp(hl * 0.95, -ht * 0.35), hp(hl * 0.95, -ht * 0.12),
                 hp(hl * 0.12, -ht * 0.25)], cc('#6a5474'))
        cv.line([hp(hl * 0.1, -ht * 0.55), hp(hl * 0.95, -ht * 0.35)], 1.6 * u, np.array([1.6, 1.05, 0.66], np.float32), 0.95)
        cv.line([hp(hl * 0.95, -ht * 0.35), hp(hl * 1.02, ht * 0.2)], 1.2 * u, cc('#ffb886'), 0.7)
        # dark underside lip + glowing lens
        cv.poly([hp(hl * 0.22, ht * 0.5), hp(hl * 0.9, ht * 0.5), hp(hl * 0.86, ht * 0.72), hp(hl * 0.26, ht * 0.72)],
                np.array([1.35, 1.1, 0.78], np.float32))
        cv.line([hp(hl * 0.22, ht * 0.5), hp(hl * 0.9, ht * 0.5)], 1.0 * u, cc('#1a1830'), 0.8)
        thk = ht * 0.6
        self.lamp = (hx - hl * 0.5, hy + thk * 1.2)
        # drop cable: from the pole, sagging to the left, leaving through the top of the frame
        y1c = px(5.95)
        cx0 = xc - half(y1c)
        # (the cables leave to the right, out of frame: the crane-down never reveals a loose end)
        cx0 = xc + half(y1c)
        ex, ey = self.X(1.2), y1c - 0.16 * H
        tt = np.linspace(0, 1, 120)
        cxs = cx0 + (ex - cx0) * tt
        cys = y1c + (ey - y1c) * tt + 0.07 * H * 4 * tt * (1 - tt)
        cv.line(np.stack([cxs, cys], 1), max(2.2 * u, 1.0), cc('#1e1c30'))
        cv.line(np.stack([cxs, cys - 1.0 * u], 1), max(0.7 * u, 0.5), cc('#ff9e70'), 0.6)
        # second, thinner cable (telecom) with a small closure box
        cys2 = y1c + 0.05 * H + (ey + 0.02 * H - y1c - 0.05 * H) * tt + 0.1 * H * 4 * tt * (1 - tt)
        cxs2 = cx0 + (self.X(1.18) - cx0) * tt
        cv.line(np.stack([cxs2, cys2], 1), max(1.6 * u, 0.8), cc('#1e1c30'))
        k = 30
        bx_, by_ = cxs2[k], cys2[k]
        cv.poly([(bx_ - 14 * u, by_ - 7 * u), (bx_ + 14 * u, by_ - 9 * u), (bx_ + 14 * u, by_ + 9 * u),
                 (bx_ - 14 * u, by_ + 11 * u)], cc('#2c2a44'))
        cv.line([(bx_ - 14 * u, by_ - 7 * u), (bx_ + 14 * u, by_ - 9 * u)], 1.2 * u, cc('#ffb27a'), 0.7)
        # ---- pole advert (white plate, blue band) and address plate: real Japanese text
        self._plate(cv, xc, half, px, 5.15, 4.05, cc('#d8d4dc'), [
            ('band', cc('#2a5aa8'), 0.0, 0.2),
            ('text', '海辺食堂', cc('#1e2a4a'), 0.24, 0.8),
            ('text', 'この先すぐ', cc('#b83838'), 0.8, 0.99),
        ])
        self._plate(cv, xc, half, px, 3.3, 2.7, cc("#2a5aa8"), [
            ('text', '汐見町二丁目', cc('#eef0f6'), 0.04, 0.99),
        ])

    def _plate(self, cv, xc, half, px, Ytop, Ybot, base, items):
        u = self.u
        yt, yb = px(Ytop), px(Ybot)
        hw = half(yt) * 0.92
        cv_rect = [(xc - hw, yt), (xc + hw, yt), (xc + hw, yb), (xc - hw, yb)]
        # the plate is in shadow (backlit): tone everything down, faint warm bounce from the road
        shade = 0.46
        cv.poly(cv_rect, base * shade)
        for it in items:
            if it[0] == 'band':
                _, c, a, b = it
                cv.poly([(xc - hw, yt + (yb - yt) * a), (xc + hw, yt + (yb - yt) * a),
                         (xc + hw, yt + (yb - yt) * b), (xc - hw, yt + (yb - yt) * b)], c * shade)
                m = text_mask('食事処', hw * 0.5, vertical=False)
                stamp(cv, m, xc, yt + (yb - yt) * (a + b) / 2, cc('#f0f0f4') * shade)
            else:
                _, txt, c, a, b = it
                span = (yb - yt) * (b - a)
                size = min(span / (len(txt) * 1.05), hw * 1.5)
                m = text_mask(txt, size, vertical=True)
                stamp(cv, m, xc, yt + (yb - yt) * a, c * (shade + 0.08), anchor='t')
        # frame edge + warm rim on the sun-side edge
        cv.line([(xc - hw, yt), (xc - hw, yb)], 1.6 * u, np.array([1.2, 0.68, 0.4], np.float32), 0.9)
        cv.line([(xc - hw, yt), (xc + hw, yt)], 1.2 * u, cc('#b08070'), 0.8)
        # mounting bands
        for yy in (yt + 4 * u, yb - 4 * u):
            cv.line([(xc - half(yy) - 2 * u, yy), (xc + half(yy) + 2 * u, yy)], 2.2 * u, cc('#3a3852'))

    # ------------------------------------------------------------------ pipe fence (bottom right)
    def _fence(self, cv):
        W, H = self.sc.W, self.sc.H
        u = self.u
        cv.zmode = ('c', 1.0 / Z_POLE)
        xa, xb = self.X(0.5), self.X(1.22)
        ya, yb = 0.905 * H + self.T, 0.94 * H + self.T
        d2 = 0.065 * H
        th = 0.016 * H

        def ry(x, off=0.0):
            return ya + (yb - ya) * (x - xa) / (xb - xa) + off
        # posts
        for x in np.arange(xa, xb, 0.13 * W):
            pw = 0.0065 * H
            cv.poly([(x - pw, ry(x) - th * 0.3), (x + pw, ry(x) - th * 0.3), (x + pw, H * 1.05 + self.T), (x - pw, H * 1.05 + self.T)],
                    cc('#2e2e48'))
            cv.line([(x - pw * 0.7, ry(x)), (x - pw * 0.7, H * 1.05 + self.T)], max(1.3 * u, 0.8), cc('#ffb27a'), 0.7)
            cv.poly([(x - pw * 1.5, ry(x) - th * 0.8), (x + pw * 1.5, ry(x) - th * 0.8), (x + pw * 1.5, ry(x) + th * 0.2),
                     (x - pw * 1.5, ry(x) + th * 0.2)], cc('#34344e'))
        # rails (galvanised pipes): cool body, hot top highlight where the low sun grazes them
        for off, tk in ((0.0, th), (d2, th * 0.8)):
            xs = np.linspace(xa - 0.01 * W, xb, 60)
            top = np.stack([xs, ry(xs, off) - tk / 2], 1)
            bot = np.stack([xs, ry(xs, off) + tk / 2], 1)[::-1]
            cv.poly(np.concatenate([top, bot]), cc('#3c3e5c'))
            cv.poly(np.concatenate([np.stack([xs, ry(xs, off) + tk * 0.05], 1),
                                    np.stack([xs, ry(xs, off) + tk / 2], 1)[::-1]]), cc('#272844'))
            # highlight: brighter toward the sun (x ~ 0.655 W)
            hot = np.exp(-((xs - self.X(0.66)) / (0.25 * W)) ** 2)
            for i in range(len(xs) - 1):
                a = 0.45 + 0.55 * hot[i]
                cv.line([(xs[i], ry(xs[i], off) - tk * 0.3), (xs[i + 1], ry(xs[i + 1], off) - tk * 0.3)],
                        max(tk * 0.22, 1.0), np.array([1.25, 0.82, 0.5], np.float32) * a, 0.95)
        self.fence_glint = (self.X(0.66), ry(self.X(0.66)) - th * 0.3)

    # ------------------------------------------------------------------ pampas grass (front)
    def _pampas(self, cv, wcv):
        W, H = self.sc.W, self.sc.H
        u = self.u
        rng = np.random.default_rng(405)
        cv.zmode = ('c', 1.0 / Z_GRASS)
        stem = cc('#2a2438')
        plume = np.array([1.08, 0.8, 0.56], np.float32)
        blade = cc('#1a1a2c')
        rim = cc('#ffb070')
        clumps = [(0.69, 0.4), (0.78, 0.36), (0.8, 0.3), (0.99, 0.31), (1.07, 0.5), (1.14, 0.4),
                  (0.19, 0.28), (-0.1, 0.36)]
        for (fx, hh) in clumps:
            bx = self.X(fx)
            by = 1.08 * H + self.T
            # long leaves first
            P.grass_tuft(cv, bx, by, hh * 0.55 * H, rng, blade, rim, lean=0.25, n=10, width=0.45, wcv=wcv,
                         rim_amt=0.8)
            for k in range(int(rng.integers(3, 6))):
                h = hh * H * rng.uniform(0.7, 1.05)
                hot = 1.0 + 0.3 * math.exp(-((fx - 0.66) / 0.18) ** 2)
                P.susuki(cv, bx + rng.normal(0, 0.012 * W), by, h, rng, stem, plume * rng.uniform(0.85, 1.05) * hot,
                         lean=rng.uniform(-0.05, 0.25), wcv=wcv)
