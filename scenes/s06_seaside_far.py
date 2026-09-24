"""Far plates for s06_seaside: distant islands (atmospheric layers), the headland with its sunlit cliff
face, tree-lined crest, sea stacks, surf line, reflection and the small white lighthouse."""
import math
import numpy as np
import cv2

from lib import core as C
import s06_seaside_paint as P


def cc(h):
    return C.hex2rgb(h)


def _smooth_noise(n, seed, sigma):
    r = np.random.default_rng(seed)
    v = r.standard_normal(n + 8 * int(sigma) + 8).astype(np.float32)
    v = cv2.GaussianBlur(v[None], (0, 0), sigmaX=sigma)[0]
    v = v[4 * int(sigma) + 4: 4 * int(sigma) + 4 + n]
    return v / (v.std() + 1e-6)


def _treeline(xs, top, rng, size, density=1.0):
    """Bumpy canopy silhouette: add rounded tree bumps on top of a profile."""
    out = top.copy()
    x = xs[0]
    while x < xs[-1]:
        r = size * rng.uniform(0.6, 1.4)
        m = np.abs(xs - x) < r
        bump = np.sqrt(np.clip(1 - ((xs[m] - x) / r) ** 2, 0, 1)) * r * rng.uniform(0.25, 0.45)
        out[m] = np.minimum(out[m], top[m] - bump)
        x += r * rng.uniform(0.7, 1.4) / density
    return out


class Far:
    def __init__(self, sc):
        self.sc = sc

    def fr(self, fx, fy):
        sc = self.sc
        return sc.ox + np.asarray(fx) * sc.W, sc.oy + np.asarray(fy) * sc.H

    # ------------------------------------------------------------------ islands
    def islands(self):
        sc = self.sc
        W, H = sc.W, sc.H
        cv = P.Canvas(sc.pw, sc.ph)
        hz = sc.hzp
        rng = np.random.default_rng(8)
        sky_h = sc.sky[int(hz) - 3]            # horizon row of the sky (pw, 3)

        def island(x0, x1, height, seed, col, rim, peak=0.4, tree=0.004, rough=0.25):
            n = 400
            xs = np.linspace(x0, x1, n)
            t = np.linspace(0, 1, n)
            prof = np.where(t < peak, (t / peak) ** 0.8, ((1 - t) / (1 - peak)) ** 1.1)
            prof = np.sin(prof * np.pi / 2) ** 1.2
            nz = _smooth_noise(n, seed, 8) * 0.12 + _smooth_noise(n, seed + 1, 2.5) * 0.04
            top = hz - height * np.clip(prof * (1 + rough * nz), 0, None)
            top = _treeline(xs, top, np.random.default_rng(seed + 3), tree * W, 1.0)
            pts = list(zip(xs, top)) + [(x1, hz + 1.5), (x0, hz + 1.5)]
            r = cv.poly_mask(pts, ss=4)
            m, bx, by = r
            h, w = m.shape
            gx, gy = cv.grid(bx, by, w, h)
            # vertical haze gradient: bottom melts into the horizon glow
            v = np.clip((hz - gy) / max(height, 1), 0, 1)
            base = col[None, None] * (0.8 + 0.2 * v[..., None])
            hzc = sky_h[np.clip(gx.astype(int), 0, sc.pw - 1)]
            c = C.lerp(hzc, base, (0.35 + 0.65 * C.smoothstep(0.0, 0.5, v))[..., None])
            # rim on the sun-facing (left) slopes: silhouette pixels whose left neighbour is empty
            ml = np.zeros_like(m)
            ml[:, 3:] = m[:, :-3]
            edge = np.clip(m - ml, 0, 1)
            c = c + (rim - c) * (edge * 0.45)[..., None]
            cv.put(r, c)

        island(0.905 * W + sc.ox, 1.16 * W + sc.ox, 0.045 * H, 11, cc('#8c6c9a'), cc('#ffc8a0'), peak=0.35)
        island(0.73 * W + sc.ox, 0.86 * W + sc.ox, 0.016 * H, 12, cc('#c893a8'), cc('#ffd4b0'), peak=0.5,
               tree=0.0025)
        island(0.82 * W + sc.ox, 0.95 * W + sc.ox, 0.026 * H, 13, cc('#b0819e'), cc('#ffcca8'), peak=0.6,
               tree=0.003)
        return cv.straight()

    # ------------------------------------------------------------------ headland + lighthouse
    def headland(self):
        sc = self.sc
        W, H = sc.W, sc.H
        cv = P.Canvas(sc.pw, sc.ph)
        rng = np.random.default_rng(21)
        hz = sc.hzp
        wl = hz + 0.034 * H
        n = 900
        fx = np.linspace(0.02, 0.455, n)
        t = (fx - fx[0]) / (fx[-1] - fx[0])
        # profile: hills on the left, saddle, plateau with the lighthouse, sheer cliff at the tip
        prof = (0.105 * (1 - C.smoothstep(0.0, 0.62, t)) ** 1.3 + 0.03
                + 0.018 * np.exp(-((t - 0.3) / 0.08) ** 2) - 0.01 * np.exp(-((t - 0.6) / 0.06) ** 2))
        prof = prof + _smooth_noise(n, 3, 10) * 0.003 + _smooth_noise(n, 4, 3) * 0.001
        xs, _ = self.fr(fx, 0)
        top = hz - prof * H
        top = top + 0.0  # plateau near the tip
        tip0 = 0.88
        top = np.where(t > tip0, top + (t - tip0) / (1 - tip0) * 0.004 * H, top)
        canopy = _treeline(xs, top, np.random.default_rng(5), 0.008 * W, 1.3)
        # clear area around the lighthouse
        lx = self.fr(0.415, 0)[0]
        near_lh = np.abs(xs - lx) < 0.025 * W
        canopy = np.where(near_lh, np.maximum(canopy, top - 0.002 * H), canopy)
        cliff_top_x = xs[-1]
        cliff = [(cliff_top_x + 0.004 * W, top[-1] + 0.006 * H), (cliff_top_x + 0.01 * W, top[-1] + 0.018 * H),
                 (cliff_top_x + 0.013 * W, wl - 0.006 * H), (cliff_top_x + 0.018 * W, wl)]
        body = list(zip(xs, canopy)) + cliff + [(xs[0], wl)]
        # -- body: hazy backlit violet with a vertical gradient (lighter/hazier toward the water)
        r = cv.poly_mask(body, ss=4)
        m, bx, by = r
        h, w = m.shape
        gx, gy = cv.grid(bx, by, w, h)
        v = np.clip((wl - gy) / (0.12 * H), 0, 1)
        c_top = cc('#4d3f78')
        c_low = cc('#8a6c9c')
        col = C.lerp(c_low, c_top, C.smoothstep(0.0, 0.7, v)[..., None])
        cv.put(r, col)
        # -- canopy texture: rows of rounded tree masses, each with a warm sunlit upper-right edge
        crng = np.random.default_rng(77)
        dark = cc('#3f3470')
        for row in range(3):
            i = int(crng.integers(0, 6))
            while i < n:
                if near_lh[i] or t[i] > 0.9:
                    i += 5
                    continue
                rr = 0.0045 * W * crng.uniform(0.7, 1.3) * (1.0 - 0.25 * row)
                x = xs[i] + crng.uniform(-0.3, 0.3) * rr
                y = canopy[i] + rr * (0.55 + 1.35 * row) + crng.uniform(0, 0.3) * rr
                if y > wl - 0.012 * H:
                    i += 6
                    continue
                ell = [(x + rr * math.cos(a), y + rr * 0.62 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 18)]
                shade = C.lerp(dark, cc('#5a4a86'), float(crng.uniform(0, 1)) * 0.6 + 0.2 * row)
                cv.poly(ell, shade, alpha=0.55)
                lit = [(x + rr * math.cos(a), y + rr * 0.62 * math.sin(a)) for a in np.linspace(-1.9, 0.25, 10)] +                       [(x + rr * 0.55 * math.cos(a), y - rr * 0.05 + rr * 0.45 * math.sin(a))
                       for a in np.linspace(0.1, -1.7, 8)]
                wa = (0.12 + 0.5 * t[i] ** 1.5) * (1.0 - 0.3 * row)
                cv.poly(lit, C.lerp(cc('#9a78b0'), cc('#f0a890'), float(t[i]) ** 2), alpha=float(wa))
                i += int(max(3, rr / (xs[1] - xs[0]) * crng.uniform(1.1, 1.8)))
        # -- the tip's cliff face catches the sun: warm salmon with vertical strata
        face = [(xs[int(n * 0.9)], top[int(n * 0.9)] + 0.002 * H)] + [(xs[-1], top[-1])] + cliff + \
               [(cliff_top_x - 0.004 * W, wl), (xs[int(n * 0.93)], top[int(n * 0.93)] + 0.02 * H)]
        rf = cv.poly_mask(face, ss=4)
        m2, fx0, fy0 = rf
        h2, w2 = m2.shape
        gx2, gy2 = cv.grid(fx0, fy0, w2, h2)
        # painted rock: vertical buttress planes (lit where the rock turns to the sun on the right, one cool
        # violet shadow value where it turns away), broken horizontal strata, a crisp lit top rim, the base
        # hazed into the sea glow
        u_ = W / 1920.0
        prof_ = _smooth_noise(w2 + 60, 9, 5.0 * u_ + 1.0)[30:30 + w2] * 1.0 + _smooth_noise(w2 + 60, 19, 1.6 * u_ + 0.5)[30:30 + w2] * 0.25
        slope_ = np.gradient(prof_)
        face_lit = C.smoothstep(-0.03, 0.03, slope_ / (np.abs(slope_).max() + 1e-6) * 0.3)[None]                    # plane faces the sun
        lit = C.smoothstep(xs[int(n * 0.9)], cliff_top_x + 0.006 * W, gx2)     # the headland turns toward the sun
        vy2 = C.smoothstep(top[-1], wl, gy2)
        # buttresses lean slightly: shear the plane pattern with height
        sh_ = np.clip((gx2 - fx0 + (gy2 - fy0) * 0.06).astype(np.int32), 0, w2 - 1)
        fl = face_lit[0][sh_]
        # strata: broken horizontal bands (value only), slightly tilted
        sy = (gy2 - fy0) + (gx2 - fx0) * 0.08
        st = np.sin(sy / (2.6 * u_ + 0.8) + 0.8 * _smooth_noise(w2, 23, 12)[None]) * 0.5 + 0.5
        strata_d = C.smoothstep(0.82, 0.95, st) * (0.5 + 0.5 * C.smoothstep(0.3, 0.7, _smooth_noise(w2, 29, 5)[None] * 0.5 + 0.5))
        c_lit, c_lit2 = cc('#f7a07a'), cc('#ffc28e')
        c_mid, c_shd = cc('#c07484'), cc('#6c548e')
        L_ = lit * (0.55 + 0.45 * fl) * (1 - 0.25 * vy2)
        fc = C.lerp(c_shd, c_mid, C.smoothstep(0.15, 0.45, L_)[..., None])
        fc = C.lerp(fc, c_lit, C.smoothstep(0.45, 0.7, L_)[..., None])
        fc = C.lerp(fc, c_lit2, (C.smoothstep(0.75, 0.95, L_) * (1 - vy2) * 0.8)[..., None])
        fc = fc * (1 - 0.18 * strata_d * (0.4 + 0.6 * L_))[..., None]
        # crisp lit rim along the top edge of the face (the plateau lip catches the sun)
        mtop = m2 > 0.5
        first = np.argmax(mtop, axis=0)
        dtop = (gy2 - fy0) - first[None]
        rim_ = C.smoothstep(2.2 * u_ + 0.6, 0.6, dtop) * (dtop >= 0) * C.smoothstep(0.2, 0.6, lit)
        fc = C.lerp(fc, cc('#ffd9a0'), (0.85 * rim_)[..., None])
        # base haze into the water (sea glow colour), the last ~2.5% of the frame height
        haze = C.smoothstep(wl - 0.03 * H, wl + 0.002 * H, gy2) * 0.75
        fc = C.lerp(fc, cc('#d09aa8'), haze[..., None])
        cv.put(rf, fc)
        # -- rim light on the crest (sun from the right, slightly behind): thin gold line on the canopy
        rim_pts = np.stack([xs, canopy], 1)
        slope = np.gradient(canopy)
        for i in range(0, n - 4, 4):
            a = np.clip(0.5 + slope[i] * 1.5, 0, 1) * (0.4 + 0.6 * t[i])
            if a < 0.05:
                continue
            seg = rim_pts[i:i + 5] + [0, 0.35 * W / 1920]
            cv.line(seg, max(0.9 * W / 1920, 0.6), cc('#ffc98c'), alpha=float(a) * 0.9)
        # -- a few sunlit tree crowns along the crest (tiny highlights)
        for i in range(0, n, 7):
            if rng.random() < 0.35 * t[i] and not near_lh[i]:
                x, y = xs[i], canopy[i] + 0.0015 * H
                rr = 0.0022 * W * rng.uniform(0.6, 1.2)
                cv.poly([(x + rr * math.cos(a), y + rr * 0.7 * math.sin(a)) for a in np.linspace(-math.pi, 0.3, 10)],
                        cc('#9a6a8a'), alpha=0.6)
        # -- sea stacks off the tip
        for (ox_, hh, ww) in ((0.022, 0.011, 0.006), (0.031, 0.006, 0.004), (0.037, 0.0035, 0.003)):
            x0 = cliff_top_x + ox_ * W
            pts = [(x0, wl), (x0 + ww * 0.2 * W, wl - hh * H), (x0 + ww * 0.55 * W, wl - hh * 1.08 * H),
                   (x0 + ww * W, wl)]
            cv.poly(pts, cc('#6a4f86'))
            cv.poly([(x0 + ww * 0.55 * W, wl - hh * 1.08 * H), (x0 + ww * W, wl), (x0 + ww * 0.7 * W, wl)],
                    cc('#e88c7c'), alpha=0.9)
        self.wl = wl
        self.cliff_x = cliff_top_x
        # -- fishing village at the foot of the headland
        import s06_seaside_village as V
        V.village(self, cv, wl)
        # -- lighthouse
        self._lighthouse(cv, lx, float(np.interp(lx, xs, top)) + 0.001 * H)
        return cv.straight()

    def _lighthouse(self, cv, x, base):
        sc = self.sc
        W, H = sc.W, sc.H
        u = W / 1920.0
        ht = 0.05 * H
        wb, wt = 0.0055 * W, 0.0042 * W
        top = base - ht
        white_s = cc('#b9aedc')
        white_l = cc('#ffe2c6')
        # keeper's house with a roof, left of the tower
        hx = x - 0.012 * W
        cv.poly([(hx - 0.007 * W, base), (hx - 0.007 * W, base - 0.008 * H), (hx + 0.006 * W, base - 0.008 * H),
                 (hx + 0.006 * W, base)], cc('#a898c8'))
        cv.poly([(hx + 0.003 * W, base), (hx + 0.003 * W, base - 0.008 * H), (hx + 0.006 * W, base - 0.008 * H),
                 (hx + 0.006 * W, base)], cc('#f0c8b0'))
        cv.poly([(hx - 0.0085 * W, base - 0.0075 * H), (hx - 0.004 * W, base - 0.014 * H),
                 (hx + 0.0045 * W, base - 0.014 * H), (hx + 0.0075 * W, base - 0.0075 * H)], cc('#6a4a78'))
        # tower body: shaded cylinder (cool on the left, warm sunlit strip on the right)
        pts = [(x - wb, base), (x - wt, top), (x + wt, top), (x + wb, base)]
        r = cv.poly_mask(pts, ss=6)
        m, bx, by = r
        h, w = m.shape
        gx, gy = cv.grid(bx, by, w, h)
        vy = np.clip((base - gy) / ht, 0, 1)
        hw = wb + (wt - wb) * vy
        nx = np.clip((gx - x) / hw, -1, 1)
        lam = C.smoothstep(0.1, 0.45, nx)
        col = C.lerp(white_s, white_l, lam[..., None])
        col = col * (1 - 0.15 * C.smoothstep(-0.2, -1.0, nx))[..., None]
        cv.put(r, col)
        # gallery (balcony) + railing
        gy0 = top
        cv.poly([(x - wt * 1.5, gy0), (x + wt * 1.5, gy0), (x + wt * 1.5, gy0 - 0.0028 * H),
                 (x - wt * 1.5, gy0 - 0.0028 * H)], cc('#5a3f6c'))
        cv.poly([(x + wt * 0.6, gy0), (x + wt * 1.5, gy0), (x + wt * 1.5, gy0 - 0.0028 * H),
                 (x + wt * 0.6, gy0 - 0.0028 * H)], cc('#e0a080'), alpha=0.8)
        # lantern room
        ly0 = gy0 - 0.0028 * H
        lh = 0.0085 * H
        cv.poly([(x - wt * 0.85, ly0), (x + wt * 0.85, ly0), (x + wt * 0.85, ly0 - lh), (x - wt * 0.85, ly0 - lh)],
                cc('#ffe6b0'))
        for k in (-0.4, 0.2):
            cv.line([(x + wt * k, ly0), (x + wt * k, ly0 - lh)], max(0.8 * u, 0.5), cc('#6a4a70'), 0.7)
        # dome (red-brown)
        d0 = ly0 - lh
        th = np.linspace(math.pi, 2 * math.pi, 20)
        dome = [(x + wt * 1.0 * math.cos(a), d0 + 0.006 * H * math.sin(a)) for a in th]
        cv.poly(dome, cc('#8a3a5a'))
        cv.poly([(x + wt * 0.2, d0), (x + wt * 1.0, d0)] +
                [(x + wt * 1.0 * math.cos(a), d0 + 0.006 * H * math.sin(a)) for a in np.linspace(0, -1.2, 8)],
                cc('#ff9a70'), alpha=0.7)
        cv.line([(x, d0 - 0.006 * H), (x, d0 - 0.0095 * H)], max(0.9 * u, 0.5), cc('#5a3a60'))
        self.lamp = (x, ly0 - lh * 0.5)

    def surf(self):
        """Foam line along the headland's waterline + sea stacks (alpha plate animated in the scene)."""
        sc = self.sc
        W, H = sc.W, sc.H
        cv = P.Canvas(sc.pw, sc.ph)
        wl = self.wl
        rng = np.random.default_rng(4)
        x0, _ = self.fr(0.05, 0)
        x1 = self.cliff_x + 0.045 * W
        xs = np.linspace(x0, x1, 400)
        for k in range(3):
            y = wl + 0.0012 * H * k + _smooth_noise(400, 30 + k, 6) * 0.0006 * H
            a = np.clip(_smooth_noise(400, 40 + k, 3) * 0.6 + 0.5, 0, 1) * (1 - 0.3 * k)
            for i in range(0, 399, 3):
                if a[i] < 0.2:
                    continue
                cv.line(np.stack([xs[i:i + 4], y[i:i + 4]], 1), max(1.1 * W / 1920 * (1 - 0.3 * k), 0.5),
                        cc('#ffe8e0'), alpha=float(a[i]) * 0.8)
        return cv.straight()

    def reflection(self, plate):
        """Mirror of a plate below its waterline (dark, soft, streaky) for the headland."""
        sc = self.sc
        H = sc.H
        wl = self.wl
        ph, pw = plate.shape[:2]
        ys = np.arange(ph, dtype=np.float32)
        yr = 2 * wl - ys
        mapy = np.tile(yr[:, None], (1, pw)).astype(np.float32)
        mapx = np.tile(np.arange(pw, dtype=np.float32)[None], (ph, 1))
        r = cv2.remap(plate, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        r[: int(wl)] = 0
        r = cv2.GaussianBlur(r, (0, 0), sigmaX=0.6, sigmaY=0.004 * H)
        fade = np.clip(1 - (ys - wl) / (0.05 * H), 0, 1)[:, None]
        r[..., 3] *= fade * 0.55
        r[..., :3] = r[..., :3] * 0.8 + cc('#3a3070') * 0.2
        return r
