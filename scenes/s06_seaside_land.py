"""Land plates for s06_seaside: road on a sea cliff (perspective-correct, per-pixel shaded asphalt with
markings, cast shadows of the guardrail), white W-beam guardrail, verge, cut slope with concrete lattice,
hillside foliage, utility poles and wires, road sign and curve mirror."""
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree

from lib import core as C
import s06_seaside_paint as P
import s06_seaside_foliage as FO
import s06_seaside_leaf as LF

HALF = 3.3          # half road width incl. shoulders (m)
RAIL = HALF + 0.3   # guardrail lateral offset


def cc(h):
    return C.hex2rgb(h)


FOL = dict(deep=cc('#0b1620'), shd=cc('#152c3a'), sky=cc('#2a4a5c'), olv=cc('#2f4638'), mid=cc('#4f5c36'),
           lit=cc('#b0703a'), hot=cc('#ffb262'),
           rim=np.array([1.7, 1.0, 0.5], np.float32))
FOL_DARK = dict(deep=cc('#0a121c'), shd=cc('#141f2e'), sky=cc('#233648'), lit=cc('#5e4a3e'), hot=cc('#c97c52'),
                rim=np.array([1.3, 0.7, 0.4], np.float32))


def road_curve(n=2400, length=420.0):
    s = np.linspace(0, length, n)
    h0, s0, k = 0.04, 16.0, 1 / 80.0
    head = h0 - k * np.clip(s - s0, 0, None) * np.clip(1.15 - (s - 60) / 200, 0.25, 1)
    ds = s[1] - s[0]
    X = -1.7 + np.concatenate([[0], np.cumsum(np.sin(head[:-1]) * ds)])
    Z = 6.0 + np.concatenate([[0], np.cumsum(np.cos(head[:-1]) * ds)])
    nx, nz = np.cos(head), -np.sin(head)     # right-hand normal (toward the sea)
    return s, X, Z, nx, nz, head


class Land:
    def __init__(self, sc):
        self.sc = sc
        W, H = sc.W, sc.H
        self.s, self.X, self.Z, self.nx, self.nz, self.head = road_curve()
        self.u = W / 1920.0
        # sun direction (world) from its screen position
        sx, sy = sc.sun_p
        self.sd = np.array([(sx - (sc.ox + W / 2)) / sc.f, (sc.hzp - sy) / sc.f, 1.0])
        # ground shadow offset per metre of height
        self.shoff = -np.array([self.sd[0], self.sd[2]]) / self.sd[1]

    # ---------------------------------------------------------------- helpers
    def at(self, s):
        """centreline point / normal at arclength s (arrays ok)."""
        s = np.asarray(s, np.float64)
        X = np.interp(s, self.s, self.X)
        Z = np.interp(s, self.s, self.Z)
        nx = np.interp(s, self.s, self.nx)
        nz = np.interp(s, self.s, self.nz)
        return X, Z, nx, nz

    def P(self, s, d, Y):
        X, Z, nx, nz = self.at(s)
        return self.sc.proj(X + nx * d, Y, Z + nz * d)

    def strip(self, s0, s1, d0, Y0, d1, Y1, n=None):
        """Screen polygon of a strip between lateral/height (d0,Y0) and (d1,Y1) over s in [s0, s1]."""
        if n is None:
            n = int(max(20, (s1 - s0) * 3))
        ss = np.linspace(s0, s1, n)
        ax, ay = self.P(ss, d0, Y0)
        bx, by = self.P(ss, d1, Y1)
        return list(zip(ax, ay)) + list(zip(bx[::-1], by[::-1]))

    def zof(self, s, d):
        X, Z, nx, nz = self.at(s)
        return Z + nz * d

    # ---------------------------------------------------------------- build
    def build(self):
        sc = self.sc
        pw, ph = sc.pw, sc.ph
        cam = (sc.hzp, sc.f, sc.hc)

        def canvas():
            c = P.Canvas(pw, ph)
            c.cam = cam
            return c
        self.road = canvas()        # verge ground + road surface
        self.road.zmode = ('Y', 0.0)
        self._verge(self.road)
        self._road_surface(self.road)
        self.grass = canvas()       # verge vegetation (sways)
        self.gw = P.Canvas(pw, ph)          # sway weights
        self._verge_plants(self.grass, self.gw)
        self.rail = canvas()
        self._street_furniture(self.rail)
        self._rail(self.rail)
        self.hill = canvas()
        self._hill(self.hill)
        self.poles = canvas()
        self._poles(self.poles)
        gw = self.gw.straight()
        return dict(road=self.road.straight(), grass=self.grass.straight(), grass_w=gw[..., 0] * gw[..., 3],
                    rail=self.rail.straight(), hill=self.hill.straight(), poles=self.poles.straight(),
                    iz_road=self.road.iz_field(1.0), iz_grass=self.grass.iz_field(6.0),
                    iz_rail=self.rail.iz_field(2.0), iz_hill=self.hill.iz_field(10.0),
                    iz_poles=self.poles.iz_field(1.5))

    def smax_visible(self):
        return 330.0

    # ---------------------------------------------------------------- verge beyond the rail
    def _verge(self, cv):
        rng = np.random.default_rng(3)
        s1 = self.smax_visible()
        ss = np.linspace(0.5, s1, 1200)
        edge = RAIL + 1.6 + 0.9 * np.sin(ss / 7.0) + 0.5 * np.sin(ss / 2.3 + 1)
        ax, ay = self.P(ss, HALF - 0.05, 0)
        X, Z, nx, nz = self.at(ss)
        bx, by = self.sc.proj(X + nx * edge, 0.1, Z + nz * edge)
        pts = list(zip(ax, ay)) + list(zip(bx[::-1], by[::-1]))
        cv.poly(pts, cc('#3e3a5c'))

    # ---------------------------------------------------------------- road surface (per pixel)
    def _road_surface(self, cv):
        sc = self.sc
        W, H = sc.W, sc.H
        s1 = self.smax_visible()
        poly = self.strip(0.3, s1, -HALF, 0, HALF, 0, n=900)
        res = cv.poly_mask(poly, ss=3)
        m, x0, y0 = res
        h, w = m.shape
        xs, ys = cv.grid(x0, y0, w, h)
        dy = np.maximum(ys - sc.hzp, 0.5)
        Zg = sc.f * sc.hc / dy
        Xg = (xs - (sc.ox + W / 2)) * Zg / sc.f
        fp = Zg / sc.f                      # metres per pixel (lateral)
        fpz = Zg * Zg / (sc.f * sc.hc)      # metres per pixel (along the view)
        # road coordinates via nearest centreline sample
        tree = cKDTree(np.stack([self.X, self.Z], 1))
        dd, idx = tree.query(np.stack([Xg.ravel(), Zg.ravel()], 1), workers=-1)
        idx = idx.reshape(h, w)
        cs = self.s[idx]
        cx, cz = self.X[idx], self.Z[idx]
        nx, nz = self.nx[idx], self.nz[idx]
        lat = (Xg - cx) * nx + (Zg - cz) * nz
        tx, tz = -nz, nx
        cs = cs + (Xg - cx) * np.sin(self.head[idx]) + (Zg - cz) * np.cos(self.head[idx])
        # footprint along the road direction ~ mix of lateral and depth footprints
        hd = self.head[idx]
        fps = np.sqrt((fp * np.sin(hd)) ** 2 + (fpz * np.cos(hd)) ** 2) + 1e-4
        fpl = np.sqrt((fp * np.cos(hd)) ** 2 + (fpz * np.sin(hd)) ** 2) + 1e-4

        def band(v, c, hw, fpv):
            # analytic box-filtered coverage of |v - c| < hw with pixel footprint fpv
            a = np.clip((v - (c - hw)) / fpv + 0.5, 0, 1)
            b = np.clip((v - (c + hw)) / fpv + 0.5, 0, 1)
            return a - b

        # ---- asphalt base: lit warm mauve, sheen toward the far distance
        lit = cc('#8e6476')
        # grain / patches in world coords, faded with footprint (no aliasing)
        rng = np.random.default_rng(11)
        tex = C.fbm(512, 512, 24, 4, seed=21)
        tex2 = C.fbm(512, 512, 5, 3, seed=22)
        tu = (lat * 12.0) % 512
        tv = (cs * 12.0) % 512
        g1 = cv2.remap(tex, tu.astype(np.float32), tv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        g2 = cv2.remap(tex2, ((lat * 2.0) % 512).astype(np.float32), ((cs * 2.0) % 512).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fadeg = np.clip(1 - fp * 8, 0, 1)
        shade = 1 + (g1 - 0.5) * 0.12 * fadeg + (g2 - 0.5) * 0.14
        # repaired patches + sealed cracks (world-space texture, 16 px per metre, tiles every 32 m)
        ct = np.zeros((512, 512), np.uint8)
        crng = np.random.default_rng(12)
        for k in range(9):
            x0_, y0_ = crng.integers(0, 512, 2)
            cv2.rectangle(ct, (int(x0_), int(y0_)), (int(x0_ + crng.integers(20, 60)), int(y0_ + crng.integers(30, 110))), 90, -1)
        for k in range(26):
            p0 = crng.uniform(0, 512, 2)
            pts = [p0]
            ang = crng.uniform(0, 2 * np.pi)
            for j in range(int(crng.integers(4, 12))):
                ang += crng.normal(0, 0.6)
                pts.append(pts[-1] + np.array([np.cos(ang), np.sin(ang)]) * crng.uniform(4, 14))
            cv2.polylines(ct, [np.array(pts, np.int32)], False, 255, 1, cv2.LINE_AA)
        ct = cv2.GaussianBlur(ct.astype(np.float32) / 255.0, (0, 0), 0.6)
        cmap = cv2.remap(ct, ((lat * 16.0) % 512).astype(np.float32), ((cs * 16.0) % 512).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fadec = np.clip(1 - fp * 14, 0, 1)
        shade = shade * (1 - 0.22 * cmap * fadec)
        # tyre-polished tracks slightly darker/glossier
        tr = sum(band(lat, c, 0.45, fpl) for c in (-2.3, -0.9, 0.9, 2.3))
        shade = shade * (1 - 0.07 * tr)
        col = lit * shade[..., None]
        # far sheen (grazing reflection of the bright sky)
        refl = sc.refl[y0:y0 + h, x0:x0 + w]
        graz = np.clip(1 - dy / (0.2 * H), 0, 1) ** 1.5
        col = col * (1 - 0.45 * graz[..., None]) + refl * (0.55 * graz[..., None])
        # ---- cast shadows of the rail beam and posts (ground polygons projected)
        shm = self._shadow_mask(x0, y0, w, h)
        shc = cc('#3c3666') * (shade[..., None] * 0.9 + 0.1)
        shc = shc * (1 - 0.3 * graz[..., None]) + refl * 0.3 * graz[..., None] * cc('#b0a0d0')
        # ---- warm backlit sheen + aggregate glints on the lit asphalt between the rail shadows (the low sun
        # skims the road from ahead: strongest toward the sun column and along the sea-side lane)
        sunx = sc.sun_p[0]
        ks = np.exp(-((xs - sunx) / (0.32 * W)) ** 2)
        side = C.smoothstep(-1.0, HALF, lat)
        near_sh = np.clip(C.blur(shm, 4.0 * self.u) * 1.6, 0, 1) * (1 - shm)     # lit asphalt next to shadow
        sheen = (0.12 + 0.3 * near_sh) * side * ks * (1 - shm) * (0.4 + 0.6 * np.clip(1 - graz * 1.5, 0, 1))
        col = col + cc('#ffa860') * sheen[..., None] * shade[..., None]
        fine = cv2.remap(C.fbm(512, 512, 70, 2, seed=23), ((lat * 30.0) % 512).astype(np.float32),
                         ((cs * 30.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        spk = C.smoothstep(0.76, 0.86, fine) * np.clip(1 - fp * 30, 0, 1) * (1 - shm) * ks * side
        col = col + np.array([1.4, 0.95, 0.55], np.float32) * (spk * (0.25 + 0.6 * near_sh))[..., None]
        col = col * (1 - shm[..., None]) + shc * shm[..., None]
        # ---- markings (worn white, lit warm / cool in shadow)
        e = band(lat, HALF - 0.45, 0.075, fpl) + band(lat, -(HALF - 0.45), 0.075, fpl)
        dash = band((cs % 10.0), 2.5, 2.5, fps)
        ctr = band(lat, 0.0, 0.075, fpl) * dash
        wear = np.clip(0.75 + 0.5 * (g1 - 0.5) * fadeg + 0.25, 0, 1)
        mk = np.clip(e + ctr, 0, 1) * wear
        wl = cc('#fff0e0') * (1 - shm[..., None]) + cc('#b8b4e6') * shm[..., None]
        wl = wl * (1 - 0.35 * graz[..., None]) + refl * 0.35 * graz[..., None]
        col = col * (1 - mk[..., None] * 0.92) + wl * mk[..., None] * 0.92
        cv.put((m, x0, y0), col)
        self.road_px = (x0, y0, w, h)

    def _shadow_mask(self, x0, y0, w, h):
        """Ground shadow of the guardrail beam (0.45..0.8 m) and posts, rasterised in the road bbox."""
        sc = self.sc
        so = self.shoff
        cvm = P.Canvas(sc.pw, sc.ph)
        s1 = self.smax_visible()
        ss = np.linspace(0.3, s1, 1600)
        X, Z, nx, nz = self.at(ss)
        bxg, bzg = X + nx * RAIL, Z + nz * RAIL
        for (h0, h1) in ((0.47, 0.8),):
            ax, ay = sc.proj(bxg + so[0] * h0, 0, bzg + so[1] * h0)
            bx, by = sc.proj(bxg + so[0] * h1, 0, bzg + so[1] * h1)
            ok = (bzg + so[1] * h1) > 1.0
            ok &= (bzg + so[1] * h0) > 1.0
            idx = np.where(ok)[0]
            if len(idx) > 2:
                pts = list(zip(ax[idx], ay[idx])) + list(zip(bx[idx][::-1], by[idx][::-1]))
                cvm.poly(pts, (1, 1, 1))
        # posts every 4 m: thin long shadows from the base to the tip
        for sp in np.arange(2.0, s1, 4.0):
            Xp, Zp, nxp, nzp = self.at(sp)
            gx, gz = Xp + nxp * RAIL, Zp + nzp * RAIL
            tx, tz = gx + so[0] * 0.8, gz + so[1] * 0.8
            if tz < 1.0:
                # clip to the part in front of the camera
                k = (gz - 1.0) / max(gz - tz, 1e-3)
                tx, tz = gx + (tx - gx) * k, gz + (tz - gz) * k
            v = np.array([tx - gx, tz - gz]); L = np.linalg.norm(v) + 1e-6
            pn = np.array([-v[1], v[0]]) / L * 0.07
            quad = [(gx + pn[0], gz + pn[1]), (tx + pn[0], tz + pn[1]), (tx - pn[0], tz - pn[1]), (gx - pn[0], gz - pn[1])]
            qx, qy = sc.proj([q[0] for q in quad], 0, [q[1] for q in quad])
            cvm.poly(list(zip(qx, qy)), (1, 1, 1))
        a = cvm.a[y0:y0 + h, x0:x0 + w]
        a = C.blur(a, 0.6 * self.u)
        return np.clip(a, 0, 1) * 0.92

    # ---------------------------------------------------------------- guardrail
    def _rail(self, cv):
        sc = self.sc
        s1 = self.smax_visible()
        post = cc('#8580a8')
        post_lit = cc('#ffcf9a')
        # posts (behind the beam) every 4 m
        for sp in np.arange(2.0, s1, 4.0)[::-1]:
            X, Z, nx, nz = self.at(sp)
            gx, gz = X + nx * (RAIL + 0.08), Z + nz * (RAIL + 0.08)
            if gz < 3:
                continue
            cv.zmode = ('c', 1.0 / gz)
            x, y0 = sc.proj(gx, 0, gz)
            _, y1 = sc.proj(gx, 0.82, gz)
            hw = sc.f * 0.07 / gz
            pts = [(x - hw, y0), (x + hw, y0), (x + hw, y1), (x - hw, y1)]
            cv.poly(pts, post)
            if hw > 0.8:
                cv.poly([(x + hw * 0.35, y0), (x + hw, y0), (x + hw, y1), (x + hw * 0.35, y1)], post_lit * 0.8, 0.7)
        # beam: W-profile stripes (heights, colour)
        stripes = [(0.46, 0.52, '#8c87b4'), (0.52, 0.6, '#b3aed2'), (0.6, 0.665, '#7e79a6'),
                   (0.665, 0.74, '#c4bfe0'), (0.74, 0.8, '#9d98c2')]
        for h0, h1, c in stripes:
            cv.zmode = ('Y', 0.5 * (h0 + h1))
            cv.poly(self.strip(0.3, s1, RAIL, h0, RAIL, h1, n=1400), cc(c))
        # hot top edge where the low sun grazes the rail
        cv.zmode = ('Y', 0.8)
        top = self.strip(0.3, s1, RAIL, 0.79, RAIL, 0.815, n=1400)
        cv.poly(top, cc('#ffd9a0'))
        # HDR specular sheen along the top edge where it lines up with the sun (+ glint anchors)
        sx_, sy_ = sc.sun_p
        ss_ = np.linspace(0.3, s1, 700)
        px_, py_ = self.P(ss_, RAIL, 0.805)
        k_ = 0.35 * np.exp(-((px_ - sx_) / (0.45 * sc.W)) ** 2) + 0.65 * np.exp(-((px_ - sx_) / (0.2 * sc.W)) ** 2)
        zz_ = self.zof(ss_, RAIL)
        hot = np.array([1.7, 1.2, 0.75], np.float32)
        for a in range(0, len(ss_) - 1, 5):
            b = min(a + 6, len(ss_))
            ka = float(k_[a:b].mean())
            if ka < 0.02 or zz_[a] < 1.2:
                continue
            cv.zmode = ('c', 1.0 / zz_[a:b].mean())
            wd = max(sc.f * 0.012 / zz_[a:b].mean(), 0.6 * self.u)
            cv.line(np.stack([px_[a:b], py_[a:b]], 1), wd, hot, alpha=min(ka * 1.2, 1.0))
        self.rail_glints = []
        rng = np.random.default_rng(5)
        for sp in np.arange(4.0, s1, 3.3):
            X, Z, nx, nz = self.at(sp)
            gz = Z + nz * RAIL
            if gz < 4:
                continue
            gx_, gy_ = sc.proj(X + nx * RAIL, 0.81, gz)
            k = float(np.exp(-((gx_ - sx_) / (0.22 * sc.W)) ** 2))
            if k > 0.12:
                self.rail_glints.append((gx_, gy_, 1.0 / gz, k, rng.uniform(0, 6.28)))

    # ---------------------------------------------------------------- hillside
    SV = 70.0     # arclength where the road disappears behind the hill

    def fr(self, fx, fy):
        """frame fractions (at cam = 0) -> plate px"""
        sc = self.sc
        return sc.ox + fx * sc.W, sc.oy + fy * sc.H

    def ridge(self):
        sc = self.sc
        vx, vy = self.P(self.SV, -HALF, 0)
        vfx, vfy = (vx - sc.ox) / sc.W, (vy - sc.oy) / sc.H
        key = np.array([(-0.14, -0.06), (-0.04, 0.07), (0.04, 0.2), (0.1, 0.33), (0.155, 0.45), (0.195, 0.535),
                        (0.225, 0.6), (vfx, vfy)])
        # smooth spline through the key points
        t = np.linspace(0, 1, len(key))
        tt = np.linspace(0, 1, 300)
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(t, key, bc_type='natural')
        pts = cs(tt)
        return np.stack(self.fr(pts[:, 0], pts[:, 1]), 1)

    def _hill(self, cv):
        sc = self.sc
        W, H = sc.W, sc.H
        rng = np.random.default_rng(17)
        rid = self.ridge()
        ss = np.linspace(0.3, self.SV, 400)
        ex, ey = self.P(ss, -HALF, 0)
        edge = np.stack([ex, ey], 1)
        bl = self.fr(-0.2, 1.3)
        tl = self.fr(-0.2, -0.1)
        poly = [tl] + [tuple(p) for p in rid] + [tuple(p) for p in edge[::-1]] + [bl]
        hm = cv.poly_mask(poly, ss=3)
        cv.zmode = ('F', self._hill_depth())
        # base: shadowed undergrowth colour
        m, x0, y0 = hm
        cv.put(hm, cc('#1d3040'))
        self.hill_mask = hm
        # ---- cut slope with concrete lattice (world coords), s < SV
        self._lattice(cv, hm)
        # ---- foliage masses on the hill above the lattice
        self._foliage(cv, hm, rid)
        # ---- weeds in the lattice cells + foreground bushes at the foot of the slope
        self._slope_weeds(cv, hm)

    def _hill_depth(self):
        """inverse-depth field of the hillside (cut slope continued upward), painted as small quads."""
        sc = self.sc
        dc = P.Canvas(sc.pw, sc.ph)
        us = np.arange(0.0, 14.01, 0.7)
        ssv = np.concatenate([np.arange(0.3, 12, 0.6), np.arange(12, self.SV + 8, 2.0)])
        for i in range(len(ssv) - 1)[::-1]:
            s0, s1 = ssv[i], ssv[i + 1]
            for j in range(len(us) - 1):
                u0, u1 = us[j], us[j + 1]
                X0, Y0, Z0 = self.lat_pt(np.array([s0, s1, s1, s0]), np.array([u0, u0, u1, u1]))
                if Z0.min() < 1.5:
                    continue
                qx, qy = sc.proj(X0, Y0, Z0)
                dc.zmode = ('c', 1.0 / Z0.mean())
                dc.iz = dc.iz if dc.iz is not None else np.zeros((dc.H, dc.W), np.float32)
                dc.poly(list(zip(qx, qy)), (0, 0, 0), ss=1)
        return dc.iz_field(3.0, fill=1.0 / 12.0)

    def _clip_put(self, cv, res, color, hm, alpha=1.0):
        if res is None:
            return
        m, x0, y0 = res
        M, hx, hy = hm
        h, w = m.shape
        sub = np.zeros_like(m)
        ys0, xs0 = max(y0, hy), max(x0, hx)
        ys1, xs1 = min(y0 + h, hy + M.shape[0]), min(x0 + w, hx + M.shape[1])
        if ys1 > ys0 and xs1 > xs0:
            sub[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = M[ys0 - hy:ys1 - hy, xs0 - hx:xs1 - hx]
        cv.put((m * sub, x0, y0), color, alpha)

    def lat_pt(self, s, u):
        """point on the cut slope: u = height up the slope (m)."""
        X, Z, nx, nz = self.at(s)
        d = -(HALF + 0.6) - u * 0.55
        return X + nx * d, u, Z + nz * d

    def _lattice(self, cv, hm):
        sc = self.sc
        UT = 4.5
        ss = np.linspace(0.3, self.SV + 5, 300)
        # slope face (grass in the cells)
        ax, ay = sc.proj(*self.lat_pt(ss, 0.0))
        bx, by = sc.proj(*self.lat_pt(ss, UT))
        face = list(zip(ax, ay)) + list(zip(bx[::-1], by[::-1]))
        r = cv.poly_mask(face, ss=3)
        m, x0, y0 = r
        h, w = m.shape
        n1 = C.fbm(w, max(h // 5, 8), max(w / 9.0, 4), 4, seed=51, aspect=False)
        n1 = cv2.resize(n1, (w, h), interpolation=cv2.INTER_CUBIC)
        n2 = C.fbm(max(w // 4, 8), max(h // 4, 8), 6, 3, seed=52)
        n2 = cv2.resize(n2, (w, h), interpolation=cv2.INTER_CUBIC)
        tex = cc('#2c3a36') * (0.72 + 0.55 * n1[..., None])
        tex = C.lerp(tex, cc('#3a4630'), (C.smoothstep(0.55, 0.8, n2) * 0.6)[..., None])
        self._clip_put(cv, r, tex, hm)
        # shading: grass in cells lit warm, gets darker/cooler with distance
        # beams along the slope (constant u) and up the slope (constant s)
        beam_lit = cc('#8a7a9c')
        beam_shd = cc('#54496e')
        bw = 0.35
        bm = np.zeros((cv.H, cv.W), np.float32)
        topm = np.zeros((cv.H, cv.W), np.float32)

        def acc(res, dst):
            if res is None:
                return res
            m_, x_, y_ = res
            sl_ = dst[y_:y_ + m_.shape[0], x_:x_ + m_.shape[1]]
            np.maximum(sl_, m_, out=sl_)
            return res
        for u in np.arange(0.0, UT + 0.01, 2.0):
            X0, Y0, Z0 = self.lat_pt(ss, u)
            X1, Y1, Z1 = self.lat_pt(ss, u + bw)
            ax, ay = sc.proj(X0, Y0, Z0)
            bx, by = sc.proj(X1, Y1, Z1)
            ok = Z0 > 1.5
            pts = list(zip(ax[ok], ay[ok])) + list(zip(bx[ok][::-1], by[ok][::-1]))
            self._clip_put(cv, acc(cv.poly_mask(pts), bm), beam_lit, hm)
            # warm light catching the upper edge of the beam
            X3, Y3, Z3 = self.lat_pt(ss, u + bw * 0.8)
            ex_, ey_ = sc.proj(X3, Y3, Z3)
            pts = list(zip(ex_[ok], ey_[ok])) + list(zip(bx[ok][::-1], by[ok][::-1]))
            self._clip_put(cv, acc(cv.poly_mask(pts), topm), cc('#f0a47e'), hm, 0.75)
            # underside shadow of the horizontal beam
            X2, Y2, Z2 = self.lat_pt(ss, u - 0.25)
            cx_, cy_ = sc.proj(X2, Y2, Z2)
            pts = list(zip(ax[ok], ay[ok])) + list(zip(cx_[ok][::-1], cy_[ok][::-1]))
            self._clip_put(cv, cv.poly_mask(pts), cc('#26303a'), hm, 0.7)
        uu = np.linspace(0, UT + bw, 30)
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            for (d0, c) in ((0.0, beam_lit), (0.3, beam_shd)):
                X0, Y0, Z0 = self.lat_pt(np.full_like(uu, sp + d0), uu)
                X1, Y1, Z1 = self.lat_pt(np.full_like(uu, sp + d0 + 0.3), uu)
                if Z0.min() < 1.5:
                    continue
                ax, ay = sc.proj(X0, Y0, Z0)
                bx, by = sc.proj(X1, Y1, Z1)
                pts = list(zip(ax, ay)) + list(zip(bx[::-1], by[::-1]))
                self._clip_put(cv, acc(cv.poly_mask(pts), bm), c, hm)
        self._lattice_finish(cv, bm, topm, r, hm)
        # gutter at the road edge
        g = self.strip(0.3, self.SV + 5, -HALF - 0.6, 0, -HALF, 0, n=300)
        self._clip_put(cv, cv.poly_mask(g), cc('#3a3050'), hm)
        g = self.strip(0.3, self.SV + 5, -HALF - 0.12, 0, -HALF, 0, n=300)
        self._clip_put(cv, cv.poly_mask(g), cc('#a08aa8'), hm, 0.8)
        # cells: vertical darkening toward the foot of the slope + grass texture
        self._cell_texture(cv, hm, UT)

    def _lattice_finish(self, cv, bm, topm, face, hm):
        """Concrete lattice paint-over: mottled / stained concrete (vertical water streaks, chips), a warm
        rim on the sun-facing edges, soft cast shadows of the beams on the grass cells (sun to the
        right -> shadows fall left), weeds at the foot of the beams."""
        sc = self.sc
        u = self.u
        M, hx, hy = hm
        clipf = np.zeros((cv.H, cv.W), np.float32)
        clipf[hy:hy + M.shape[0], hx:hx + M.shape[1]] = M
        ys_, xs_ = np.nonzero(bm > 0.01)
        if not len(ys_):
            return
        pad = int(20 * u) + 4
        y0, y1 = max(ys_.min() - pad, 0), min(ys_.max() + pad, cv.H)
        x0, x1 = max(xs_.min() - pad, 0), min(xs_.max() + pad, cv.W)
        h, w = y1 - y0, x1 - x0
        B = bm[y0:y1, x0:x1] * clipf[y0:y1, x0:x1]
        rgb = cv.rgb[y0:y1, x0:x1]
        # --- cast shadow of the beams on the cells (offset away from the sun, soft)
        fm, fx0, fy0 = face
        F = np.zeros((cv.H, cv.W), np.float32)
        F[fy0:fy0 + fm.shape[0], fx0:fx0 + fm.shape[1]] = fm
        F = F[y0:y1, x0:x1] * clipf[y0:y1, x0:x1]
        d = 7.0 * u
        Msh = np.float32([[1, 0, -d], [0, 1, d * 0.55]])
        sh = cv2.warpAffine(B, Msh, (w, h))
        sh = cv2.GaussianBlur(sh, (0, 0), 2.2 * u) * (1 - B) * F
        rgb *= (1 - 0.45 * sh)[..., None]
        # --- concrete: mottling, vertical water stains, chips; darker toward the foot of the slope
        n1 = C.fbm(w, h, max(w / (60 * u), 3), 4, seed=91)
        st = cv2.resize(C.fbm(max(w // 3, 8), max(h // 24, 4), max(w / (14 * u) / 3, 3), 3, seed=92, aspect=False),
                        (w, h), interpolation=cv2.INTER_CUBIC)
        stain = C.smoothstep(0.52, 0.78, st)
        rng = np.random.default_rng(93)
        chips = (rng.random((h, w)) > 0.994).astype(np.float32)
        chips = cv2.GaussianBlur(chips, (0, 0), 0.8 * u) * 3.0
        tex = (0.86 + 0.28 * n1) * (1 - 0.28 * stain) * (1 - 0.35 * np.clip(chips, 0, 1))
        # moss / grime creeping up from the cells: greener-darker at the bottom edge of every beam
        und = np.clip(B - cv2.warpAffine(B, np.float32([[1, 0, 0], [0, 1, -3.5 * u]]), (w, h)), 0, 1)
        k = B[..., None]
        rgb[:] = rgb * (1 - k) + rgb * tex[..., None] * k
        grime = C.hex2rgb('#3c4a3c')
        rgb[:] = rgb + (grime * rgb.mean(-1, keepdims=True) * 1.3 - rgb) * (und * 0.45 * (0.5 + n1))[..., None]
        # --- warm rim on the sun-facing edges of the beams (thin, broken by the stains)
        L = np.float32([[1, 0, -2.2 * u], [0, 1, 0.8 * u]])
        rim = np.clip(B - cv2.warpAffine(B, L, (w, h)), 0, 1) * (0.55 + 0.45 * (1 - stain))
        rim = np.maximum(rim, topm[y0:y1, x0:x1] * clipf[y0:y1, x0:x1] * 0.35)
        rc = np.array([1.25, 0.72, 0.45], np.float32)
        rgb[:] = rgb + (rc - rgb) * (np.clip(rim, 0, 1) * 0.7)[..., None]
        cv.rgb[y0:y1, x0:x1] = rgb
        # --- weeds at the foot of the beams
        rng = np.random.default_rng(94)
        dark = cc('#16202a')
        rimc = cc('#ffab66')
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            for _ in range(int(rng.integers(1, 4))):
                X, Y, Z = self.lat_pt(sp + rng.uniform(-0.3, 0.6), rng.uniform(0.0, 0.2))
                if Z < 3:
                    continue
                bx, by = sc.proj(X, Y, Z)
                ph = sc.f * rng.uniform(0.25, 0.6) / Z
                if ph < 2:
                    continue
                P.grass_tuft(cv, bx, by, ph, rng, dark, rimc, lean=0.2, n=int(rng.integers(5, 10)), rim_amt=0.6)

    def _foliage(self, cv, hm, rid):
        """hillside canopy: clustered leaf masses (flat values, scalloped borders, warm rim on the
        sun-facing flanks), back to front; ridge masses break the skyline."""
        sc = self.sc
        W, H = sc.W, sc.H
        u = self.u
        rng = np.random.default_rng(23)
        ss = np.linspace(0.3, self.SV + 5, 300)
        tx, ty = sc.proj(*self.lat_pt(ss, 4.9))
        M, hx, hy = hm
        k = max(int(0.012 * W), 1)
        clipm = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        clip = (clipm, hx, hy)
        fill, ridge = [], []
        n = len(rid)
        for i in range(0, n, 5):
            x, y = rid[i]
            f = i / n
            r = (0.07 * (1 - f) ** 1.2 + 0.011) * W * rng.uniform(0.55, 1.45)
            ridge.append((x - r * 0.25 * rng.uniform(0.3, 1.0), y + r * 0.25, r, f))
        for kk in range(120):
            j = rng.integers(0, len(tx))
            x, y = tx[j], ty[j]
            f = j / len(tx)
            up = rng.uniform(0, 1) ** 0.7
            x = x - up * rng.uniform(0.0, 0.25) * W * (1 - f)
            y = y - up * rng.uniform(0.0, 0.5) * H * (1 - f) ** 1.2
            r = (0.055 * (1 - f) ** 1.3 + 0.01) * W * rng.uniform(0.45, 1.5) ** 1.3
            fill.append((x, y, r, f))
        # back to front: higher (farther up the slope) first, then lower / nearer
        fill.sort(key=lambda c: c[1] + c[2] * 0.5)
        for (x, y, r, f) in fill:
            LF.clump(cv, x, y, r, rng, FOL, L=(1.0, -0.35), lz=-0.3, flat=0.75, leaf=0.11, clip=clip, unit=u,
                    rim=0.85, lit_bias=-0.24 + 0.1 * f, hot_amt=0.7)
        ridge.sort(key=lambda c: c[2])
        for (x, y, r, f) in ridge:
            LF.clump(cv, x, y, r, rng, FOL, L=(1.0, -0.4), lz=-0.25, flat=0.75, leaf=0.1, unit=u, rim=1.0,
                    lit_bias=-0.17, hot_amt=0.3)

    # ---------------------------------------------------------------- verge plants
    def _verge_plants(self, cv, wcv):
        sc = self.sc
        rng = np.random.default_rng(31)
        items = []
        for sp in np.arange(1.0, 150.0, 0.35):
            X, Z, nx, nz = self.at(sp)
            edge = RAIL + 1.6 + 0.9 * np.sin(sp / 7.0) + 0.5 * np.sin(sp / 2.3 + 1)
            k = 2 if sp < 40 else 1
            for _ in range(k):
                if rng.random() < 0.25 + 0.5 * np.exp(-sp / 60):
                    d = rng.uniform(RAIL + 0.25, edge + 0.2)
                    gz = Z + nz * d
                    if gz < 3:
                        continue
                    kind = 'susuki' if rng.random() < 0.3 else 'tuft'
                    hgt = rng.uniform(1.1, 1.9) if kind == 'susuki' else rng.uniform(0.3, 0.9)
                    items.append((gz, X + nx * d, kind, hgt))
            # low shrubs along the cliff edge
            if rng.random() < 0.08:
                d = edge + rng.uniform(-0.3, 0.3)
                items.append((Z + nz * d, X + nx * d, 'shrub', rng.uniform(0.6, 1.3)))
        items.sort(key=lambda it: -it[0])
        stem = cc('#3a3450')
        plume = cc('#ffd8a4')
        dark = cc('#2c2c46')
        rim = cc('#ffb070')
        pal = dict(shd=cc('#181a2e'), mid=cc('#262840'), lit=cc('#4a4054'), rim=cc('#ffa868'))
        for gz, gx, kind, hgt in items:
            cv.zmode = ('c', 1.0 / gz)
            bx, by = sc.proj(gx, 0, gz)
            ph = sc.f * hgt / gz
            if ph < 1.2:
                continue
            if kind == 'susuki':
                P.susuki(cv, bx, by, ph, rng, stem, plume * (0.8 + 0.3 * min(1, ph / 60)), lean=0.12, wcv=wcv)
            elif kind == 'tuft':
                P.grass_tuft(cv, bx, by, ph, rng, dark, rim, lean=0.1, wcv=wcv, n=int(rng.integers(6, 14)))
            else:
                r = ph * 0.7
                LF.clump(cv, bx, by - r * 0.45, r, rng, FOL_DARK, L=(0.9, -0.5), lz=-0.3, flat=0.7,
                        leaf=0.16, unit=self.u, rim=0.8, hot_amt=0.0, lit_bias=-0.25)

    # ---------------------------------------------------------------- utility poles + wires
    def pole_top(self, sp, dx, Y):
        X, Z, nx, nz = self.at(sp)
        d = -(HALF + 0.45) + dx
        return X + nx * d, Y, Z + nz * d

    def _poles(self, cv):
        sc = self.sc
        u = self.u
        poles = [9.0, 42.0, 75.0]
        con_s = cc('#3a3452')
        con_l = cc('#f09a70')
        for sp in poles[:2][::-1]:
            X, Z, nx, nz = self.at(sp)
            d = -(HALF + 0.45)
            gx, gz = X + nx * d, Z + nz * d
            cv.zmode = ('c', 1.0 / gz)
            bx, by = sc.proj(gx, -0.5, gz)
            tx, ty = sc.proj(gx, 11.5, gz)
            wb = sc.f * 0.17 / gz
            wt = sc.f * 0.12 / gz
            pts = [(bx - wb, by), (tx - wt, ty), (tx + wt, ty), (bx + wb, by)]
            r = cv.poly_mask(pts, ss=4)
            m, x0, y0 = r
            h, w = m.shape
            gxs, gys = cv.grid(x0, y0, w, h)
            vy = np.clip((by - gys) / max(by - ty, 1), 0, 1)
            cx = bx + (tx - bx) * vy
            hw = wb + (wt - wb) * vy
            nn = np.clip((gxs - cx) / hw, -1, 1)
            col = C.lerp(con_s, con_l, C.smoothstep(0.6, 0.9, nn)[..., None])
            col = col * (1 - 0.25 * C.smoothstep(0.0, -1.0, nn))[..., None]
            cv.put(r, col)
            # step bolts
            for k in range(int((11.5 - 2.0) / 0.45)):
                Y = 2.0 + k * 0.45
                side = -1 if k % 2 == 0 else 1
                px, py = sc.proj(gx, Y, gz)
                hw_ = wb + (wt - wb) * (Y + 0.5) / 12
                L = sc.f * 0.18 / gz
                cv.line([(px + side * hw_, py), (px + side * (hw_ + L), py)], max(sc.f * 0.02 / gz, 0.5),
                        cc('#3a3656'))
            # crossarms with insulators
            for Y, span in ((10.3, 1.0), (9.5, 0.8)):
                ax, ay = sc.proj(*self.pole_top(sp, -span, Y))
                bx2, by2 = sc.proj(*self.pole_top(sp, span, Y))
                th = max(sc.f * 0.09 / gz, 0.7)
                cv.line([(ax, ay), (bx2, by2)], th, cc('#34304e'))
                cv.line([(ax, ay - th * 0.3), (bx2, by2 - th * 0.3)], max(th * 0.3, 0.5), cc('#ffb07a'), 0.7)
                for dxi in (-span * 0.9, 0.0, span * 0.9):
                    ix, iy = sc.proj(*self.pole_top(sp, dxi, Y + 0.12))
                    rr = max(sc.f * 0.06 / gz, 0.7)
                    cv.poly([(ix - rr, iy + rr), (ix - rr * 0.7, iy - rr * 1.6), (ix + rr * 0.7, iy - rr * 1.6),
                             (ix + rr, iy + rr)], cc('#5a5078'))
            # clamp + insulator where the low (7.2 m) line is dead-ended on the pole
            cx_, cy_ = sc.proj(*self.pole_top(sp, 0.0, 7.2))
            rr = max(sc.f * 0.07 / gz, 0.8)
            cv.poly([(cx_ - rr * 1.6, cy_ - rr * 0.9), (cx_ + rr * 1.6, cy_ - rr * 0.9), (cx_ + rr * 1.6, cy_ + rr * 0.9),
                     (cx_ - rr * 1.6, cy_ + rr * 0.9)], cc('#34304e'))
            cv.line([(cx_ - rr * 1.6, cy_ - rr * 0.9), (cx_ + rr * 1.6, cy_ - rr * 0.9)], max(rr * 0.4, 0.5), cc('#ffb07a'), 0.7)
            if sp > 20:
                ox_, oy_ = sc.proj(*self.pole_top(sp, 0.35, 7.6))
                rw, rh = sc.f * 0.28 / gz, sc.f * 0.9 / gz
                cv.poly([(ox_ - rw, oy_), (ox_ - rw, oy_ - rh), (ox_ + rw, oy_ - rh), (ox_ + rw, oy_)], cc('#4c466c'))
                cv.poly([(ox_ + rw * 0.4, oy_), (ox_ + rw * 0.4, oy_ - rh), (ox_ + rw, oy_ - rh), (ox_ + rw, oy_)],
                        cc('#d89a80'), 0.8)
        # wires: between consecutive poles, and from the first pole back over the camera
        self.wire_glints = []
        self.wires = []
        spans = [(-30.0, 9.0), (9.0, 42.0), (42.0, 75.0)]
        wires = [(-0.9, 10.35, 0.55), (0.0, 10.35, 0.6), (0.9, 10.35, 0.55), (-0.7, 9.55, 0.7), (0.7, 9.55, 0.7),
                 (0.0, 7.2, 0.9)]
        for (s0, s1) in spans:
            for (dx, Y, sag) in wires:
                tt = np.linspace(0, 1, 700)
                A = np.array(self.pole_top(s0, dx, Y))
                B = np.array(self.pole_top(s1, dx, Y))
                Xw = A[0] + (B[0] - A[0]) * tt
                Zw = A[2] + (B[2] - A[2]) * tt
                Yw = Y - sag * 4 * tt * (1 - tt) * (abs(s1 - s0) / 33.0) ** 2 * 1.4
                ok = Zw > 1.5
                if ok.sum() < 3:
                    continue
                px, py = sc.proj(Xw[ok], Yw[ok], Zw[ok])
                zz = Zw[ok]
                # wires are NOT painted into the plate (a per-pixel depth warp kinks thin lines that cross
                # other depths): they are stored in 3D and drawn per frame as clean catenaries
                dd = np.hypot(px - sc.sun_p[0], py - sc.sun_p[1]) / sc.W
                kk = 0.25 + 0.9 * np.exp(-(dd / 0.3) ** 2)
                self.wires.append((px.astype(np.float64), py.astype(np.float64), zz.astype(np.float64),
                                   np.clip(kk, 0, 1)))
                j = int(np.argmin(dd))
                if dd[j] < 0.4 and 0 < j < len(px) - 1:
                    self.wire_glints.append((px[j], py[j], 1.0 / zz[j], float(np.exp(-(dd[j] / 0.25) ** 2)),
                                             float(dx * 3 + Y)))

    def _cell_texture(self, cv, hm, UT):
        """Darken the lattice toward its foot (occlusion by weeds) with stacked soft strips."""
        ss = np.linspace(0.3, self.SV + 5, 300)
        for (u0, u1, a) in ((0.0, 0.6, 0.45), (0.0, 1.3, 0.3), (0.0, 2.2, 0.2), (0.0, 3.2, 0.12)):
            ax, ay = self.sc.proj(*self.lat_pt(ss, u0))
            bx, by = self.sc.proj(*self.lat_pt(ss, u1))
            pts = list(zip(ax, ay)) + list(zip(bx[::-1], by[::-1]))
            self._clip_put(cv, cv.poly_mask(pts), cc('#0e1826'), hm, a)

    def _slope_weeds(self, cv, hm):
        sc = self.sc
        W, H = sc.W, sc.H
        rng = np.random.default_rng(77)
        dark = cc('#1a2432')
        rim = cc('#ffab66')
        items = []
        for sp in np.arange(0.5, self.SV, 1.0):
            for uu in (0.25, 2.25, 4.25):
                if rng.random() < 0.55:
                    X, Y, Z = self.lat_pt(sp + rng.uniform(0.3, 1.7), uu + rng.uniform(0, 0.3))
                    if Z > 3:
                        items.append((Z, X, Y, rng.uniform(0.4, 1.1), 'tuft'))
            if rng.random() < 0.35:
                X, Y, Z = self.lat_pt(sp, 0.0)
                if Z > 3:
                    items.append((Z, X, Y, rng.uniform(0.6, 1.4), 'bush'))
        items.sort(key=lambda it: -it[0])
        pal = dict(shd=cc('#0c1622'), mid=cc('#182634'), lit=cc('#34483e'), rim=cc('#ffa864'))
        final = cv
        cv = P.Canvas(final.W, final.H)
        for Z, X, Y, hgt, kind in items:
            bx, by = sc.proj(X, Y, Z)
            ph = sc.f * hgt / Z
            if ph < 1.5:
                continue
            if kind == 'tuft':
                tmp = P.Canvas(cv.W, cv.H) if False else None
                P.grass_tuft(cv, bx, by, ph, rng, dark, rim, lean=0.15, n=int(rng.integers(6, 12)), rim_amt=0.55)
            else:
                LF.clump(cv, bx, by - ph * 0.3, ph * 0.6, rng, FOL_DARK, L=(1.0, -0.4), lz=-0.3, flat=0.7,
                        leaf=0.14, unit=self.u, rim=0.8, hot_amt=0.0, lit_bias=-0.25)
        # clip the slope weeds to the hill silhouette (they must not float over the sea near the bend)
        M, hx, hy = hm
        clip = np.zeros((cv.H, cv.W), np.float32)
        clip[hy:hy + M.shape[0], hx:hx + M.shape[1]] = M
        clip = cv2.dilate(clip, np.ones((5, 5), np.uint8))
        final.rgb = cv.rgb * clip[..., None] + final.rgb * (1 - cv.a * clip)[..., None]
        final.a = cv.a * clip + final.a * (1 - cv.a * clip)
        cv = final

    # ---------------------------------------------------------------- curve mirror + warning sign
    def _street_furniture(self, cv):
        sc = self.sc
        u = self.u
        # curve mirror (orange pole, round convex mirror) behind the rail, outside of the bend
        for (sp, dd) in ((27.0, RAIL + 0.55),):
            X, Z, nx, nz = self.at(sp)
            gx, gz = X + nx * dd, Z + nz * dd
            cv.zmode = ('c', 1.0 / gz)
            self.mirror_iz = 1.0 / gz
            bx, by = sc.proj(gx, 0, gz)
            tx, ty = sc.proj(gx, 2.9, gz)
            pw_ = max(sc.f * 0.05 / gz, 0.8)
            cv.poly([(bx - pw_, by), (tx - pw_, ty), (tx + pw_, ty), (bx + pw_, by)], cc('#c8603a'))
            cv.poly([(bx + pw_ * 0.2, by), (tx + pw_ * 0.2, ty), (tx + pw_, ty), (bx + pw_, by)], cc('#ffb078'), 0.9)
            R = sc.f * 0.42 / gz
            mx, my = tx, ty - R * 0.6
            th = np.linspace(0, 2 * math.pi, 40)
            ring = [(mx + R * 1.12 * math.cos(a), my + R * 1.12 * math.sin(a)) for a in th]
            cv.poly(ring, cc('#d0643c'))
            disc = [(mx + R * math.cos(a), my + R * math.sin(a)) for a in th]
            r = cv.poly_mask(disc, ss=4)
            m, x0, y0 = r
            h, w = m.shape
            gxs, gys = cv.grid(x0, y0, w, h)
            v = np.clip((gys - (my - R)) / (2 * R), 0, 1)
            # convex mirror: compressed sky (warm at the bottom) + the dark road band
            col = C.lerp(cc('#8a86d8'), cc('#ffc890'), (v ** 1.5)[..., None])
            col = C.lerp(col, cc('#3a3050'), C.smoothstep(0.68, 0.74, v)[..., None])
            hl = np.exp(-(((gxs - (mx + R * 0.35)) / (R * 0.25)) ** 2 + ((gys - (my - R * 0.3)) / (R * 0.4)) ** 2))
            col = col + hl[..., None] * cc('#fff0d0') * 0.6
            cv.put(r, col)
            self.mirror_glint = (mx + R * 0.35, my - R * 0.3, R)
        # yellow diamond curve-warning sign
        for (sp, dd) in ((16.0, RAIL + 0.45),):
            X, Z, nx, nz = self.at(sp)
            gx, gz = X + nx * dd, Z + nz * dd
            cv.zmode = ('c', 1.0 / gz)
            bx, by = sc.proj(gx, 0, gz)
            tx, ty = sc.proj(gx, 2.0, gz)
            pw_ = max(sc.f * 0.035 / gz, 0.7)
            cv.poly([(bx - pw_, by), (tx - pw_, ty), (tx + pw_, ty), (bx + pw_, by)], cc('#6a6488'))
            S_ = sc.f * 0.42 / gz
            cx_, cy_ = tx, ty - S_ * 0.9
            dia = [(cx_, cy_ - S_), (cx_ + S_, cy_), (cx_, cy_ + S_), (cx_ - S_, cy_)]
            cv.poly([(x, y) for x, y in [(cx_, cy_ - S_ * 1.08), (cx_ + S_ * 1.08, cy_), (cx_, cy_ + S_ * 1.08),
                                         (cx_ - S_ * 1.08, cy_)]], cc('#3a3450'))
            cv.poly(dia, cc('#d8a038'))           # backlit yellow face (in shade)
            # curved arrow (bend to the left)
            tt = np.linspace(0, 1, 20)
            ax_ = cx_ + S_ * 0.25 - S_ * 0.35 * np.sin(tt * 1.6) ** 1.0
            ay_ = cy_ + S_ * 0.55 - tt * S_ * 1.0
            cv.line(np.stack([ax_, ay_], 1), max(S_ * 0.14, 0.8), cc('#1e1a28'))
            hx_, hy_ = ax_[-1], ay_[-1]
            cv.poly([(hx_ - S_ * 0.28, hy_ + S_ * 0.05), (hx_ + S_ * 0.1, hy_ - S_ * 0.25),
                     (hx_ + S_ * 0.2, hy_ + S_ * 0.12)], cc('#1e1a28'))
            # sun catching the right edge of the sign plate
            cv.line([(cx_, cy_ - S_), (cx_ + S_, cy_), (cx_, cy_ + S_)], max(S_ * 0.05, 0.6), cc('#ffd9a0'), 0.9)
