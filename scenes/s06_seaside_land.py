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
import s06_seaside_tree as TR
import s06_seaside_tree2 as T2
import s06_seaside_tree3 as T3
import s06_seaside_tree4 as T4
import s06_seaside_tree5 as T5
import s06_seaside_tree6 as T6
import s06_seaside_tree7 as T7
import s06_seaside_tree8 as T8
import s06_seaside_tree9 as T9
import s06_seaside_tree10 as T10  # noqa: F401
import s06_seaside_tree11 as T11

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
        self.tree = canvas()        # the big hillside crown (own plate: it sways)
        self.tw = P.Canvas(pw, ph)          # its sway weights
        self._hill(self.hill)
        self.poles = canvas()
        self._poles(self.poles)
        gw = self.gw.straight()
        return dict(road=self.road.straight(), grass=self.grass.straight(), grass_w=gw[..., 0] * gw[..., 3],
                    rail=self.rail.straight(), hill=self.hill.straight(), poles=self.poles.straight(),
                    tree=self.tree.straight(),
                    tree_w=np.dstack([self.tw.straight()[..., 3], self.tw.straight()[..., 0]]).astype(np.float32),
                    iz_tree=self.tree.iz_field(10.0),
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
        res = cv.poly_mask(pts, ss=3)
        if res is None:
            return
        m, x0, y0 = res
        h, w = m.shape
        # dark backlit verge: grass texture in cool shade, a warm lip along the cliff edge toward the sun path
        n1 = C.fbm(w, h, max(w / (40 * self.u), 3), 4, seed=33)
        n2 = C.fbm(w, h, max(w / (8 * self.u), 3), 2, seed=34)
        col = cc('#27233c') * (0.8 + 0.4 * n1[..., None]) * (0.92 + 0.16 * n2[..., None])
        xs, ys = cv.grid(x0, y0, w, h)
        ks = np.exp(-((xs - self.sc.sun_p[0]) / (0.25 * self.sc.W)) ** 2)
        # outer edge: a thin band just inside the far polygon edge
        mb = cv2.erode((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(int(2 * self.u), 1))
        lip = np.clip(m - cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 1.5 * self.u), 0, 1)
        lip = lip * (ys < np.interp(xs, bx, by) + 4 * self.u) * C.smoothstep(0.45, 0.6, n2)
        col = col + (np.array([1.5, 0.85, 0.45], np.float32) - col) * (lip * ks * 0.8)[..., None]
        cv.put(res, col)

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
        lit = cc('#9a6a6c')
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
        shade = shade * (1 - 0.34 * cmap * fadec)
        # tyre-polished tracks slightly darker/glossier
        tr = sum(band(lat, c, 0.45, fpl) for c in (-2.3, -0.9, 0.9, 2.3))
        shade = shade * (1 - 0.07 * tr)
        # oil drip strip down the middle of each lane, pale dust along the kerbs
        oil = sum(band(lat, c, 0.3, fpl) for c in (-1.6, 1.6)) * (0.6 + 0.8 * g2)
        shade = shade * (1 - 0.09 * oil)
        dust = C.smoothstep(HALF - 0.9, HALF - 0.15, np.abs(lat)) * (0.5 + 0.5 * g2)
        shade = shade * (1 + 0.1 * dust)
        # fine sealed cracks (tar snakes, 48 px per metre, tiles every ~21 m): dark, glossy
        ct2 = np.zeros((1024, 1024), np.uint8)
        for k in range(70):
            p0 = crng.uniform(0, 1024, 2)
            ang = crng.uniform(0, 2 * np.pi)
            stack = [(p0, ang, int(crng.integers(6, 18)))]
            while stack:
                q, a_, n_ = stack.pop()
                pts = [q]
                for j in range(n_):
                    a_ += crng.normal(0, 0.45)
                    pts.append(pts[-1] + np.array([np.cos(a_), np.sin(a_)]) * crng.uniform(5, 16))
                    if crng.random() < 0.08 and len(stack) < 6:
                        stack.append((pts[-1], a_ + crng.choice([-1, 1]) * crng.uniform(0.6, 1.3), n_ // 2))
                cv2.polylines(ct2, [np.array(pts, np.int32)], False, 255, int(crng.integers(1, 3)), cv2.LINE_AA)
        ct2 = cv2.GaussianBlur(ct2.astype(np.float32) / 255.0, (0, 0), 0.5)
        cmap2 = cv2.remap(ct2, ((lat * 48.0) % 1024).astype(np.float32), ((cs * 48.0) % 1024).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fade2 = np.clip(1.6 - fp * 60, 0, 1)
        cmap2 = cmap2 * fade2 * (1 - 0.8 * C.smoothstep(HALF - 0.6, HALF, np.abs(lat)))
        shade = shade * (1 - 0.48 * cmap2)
        # ---- worn country-road asphalt (suzume_01): patch repairs, tar seams, tyre-wear, sand drift
        prng = np.random.default_rng(71)
        nearw = np.clip(1.2 - fp * 25, 0, 1)
        patch = np.zeros_like(lat, np.float32)
        for k in range(26):
            s0_ = prng.uniform(4, 120) ** 1.0
            l0_ = prng.uniform(-HALF + 0.3, HALF - 0.3)
            ls_, ll_ = prng.uniform(0.8, 3.5), prng.uniform(0.5, 1.8)
            if prng.random() < 0.3:          # long trench reinstatement strip across the lane
                ll_ = HALF
                l0_ = prng.choice([-1, 1]) * HALF * 0.5
                ls_ = prng.uniform(0.5, 0.9)
            pk = band(cs, s0_, ls_ / 2, fps) * band(lat, l0_, ll_ / 2, fpl)
            val = prng.choice([-0.26, -0.17, 0.12])
            patch = patch * (1 - pk) + val * pk
        # sealed seam around every patch edge: dark glossy line
        pe = np.abs(patch)
        pe = np.clip(pe - cv2.erode(pe, np.ones((3, 3), np.uint8)), 0, 1) * 5.0 * nearw
        shade = shade * (1 + patch) * (1 - 0.35 * np.clip(pe, 0, 1))
        # longitudinal construction joint near the centre line and transverse joints every ~14 m
        seam = band(lat, 0.18, 0.03, fpl) + band(cs % 14.0, 7.0, 0.035, fps) * (np.abs(lat) < HALF - 0.2)
        shade = shade * (1 - 0.25 * np.clip(seam, 0, 1) * nearw)
        # tyre-wear: two darker, polished wheel paths in each lane (soft profile)
        wp = sum(np.exp(-((lat - c) / 0.38) ** 2) for c in (-2.35, -0.95, 0.95, 2.35))
        shade = shade * (1 - 0.17 * wp * (0.7 + 0.6 * g2))
        # sand / fine gravel drifted against both kerbs (pale, warm, broken)
        dr = C.smoothstep(HALF - 0.75, HALF - 0.1, np.abs(lat)) * C.smoothstep(0.35, 0.65, g1 * 0.5 + g2 * 0.7 - 0.1)
        dr = dr * (1.3 - 0.3 * np.sign(lat))           # heavier on the hill side (washed off the slope)
        shade = shade * (1 + 0.28 * dr)
        grit = C.smoothstep(0.7, 0.8, fine_ := cv2.remap(C.fbm(512, 512, 90, 2, seed=29), ((lat * 40.0) % 512).astype(np.float32),
                                                        ((cs * 40.0) % 512).astype(np.float32), cv2.INTER_LINEAR,
                                                        borderMode=cv2.BORDER_WRAP)) * nearw
        shade = shade * (1 + 0.2 * grit * dr) * (1 - 0.12 * grit * (1 - dr) * 0.5)
        # fine painted grit: dark aggregate specks + pale stone chips (world space, faded with footprint)
        gr1 = cv2.remap(C.fbm(512, 512, 200, 1, seed=31), ((lat * 70.0) % 512).astype(np.float32),
                        ((cs * 70.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        gfade = np.clip(1.3 - fp * 45, 0, 1)
        shade = shade * (1 - 0.16 * C.smoothstep(0.66, 0.74, gr1) * gfade
                         + 0.12 * C.smoothstep(0.36, 0.28, gr1) * gfade)
        col = lit * shade[..., None]
        # ---- manhole covers (round cast iron, rim + chequer) and kerb-side drain grates
        mh = np.zeros_like(lat, np.float32)
        mring = np.zeros_like(lat, np.float32)
        for (s_m, l_m) in ((17.0, 1.0), (27.0, -1.35), (52.0, 0.9)):
            rr_ = np.sqrt(((cs - s_m) / 1.0) ** 2 + ((lat - l_m) / 1.0) ** 2)
            fpr = np.maximum(fps, fpl)
            mh = np.maximum(mh, np.clip((0.42 - rr_) / fpr + 0.5, 0, 1))
            mring = np.maximum(mring, band(rr_, 0.42, 0.045, fpr))
        chq = (np.sin(cs * 60.0) * np.sin(lat * 60.0) > 0).astype(np.float32) * np.clip(1 - fp * 40, 0, 1)
        mcol = cc('#4a3a4c') * (1 - 0.18 * chq)[..., None]
        col = col * (1 - mh[..., None]) + mcol * mh[..., None]
        col = col * (1 - 0.5 * mring[..., None])
        dg = np.zeros_like(lat, np.float32)
        slots = np.zeros_like(lat, np.float32)
        for s_d in (8.0, 19.0, 31.0, 46.0, 64.0, 88.0):
            b_ = band(cs, s_d, 0.45, fps) * band(lat, -HALF + 0.32, 0.22, fpl)
            dg = np.maximum(dg, b_)
            slots = np.maximum(slots, b_ * (np.sin((cs - s_d) * 38.0) > 0.2) * np.clip(1 - fp * 40, 0, 1))
        col = col * (1 - dg[..., None]) + cc('#3a2e40') * dg[..., None]
        col = col * (1 - 0.6 * slots[..., None])
        # far sheen (grazing reflection of the bright sky)
        refl = sc.refl[y0:y0 + h, x0:x0 + w]
        graz = np.clip(1 - dy / (0.2 * H), 0, 1) ** 1.5
        col = col * (1 - 0.45 * graz[..., None]) + refl * (0.55 * graz[..., None])
        # ---- cast shadows of the rail beam and posts (ground polygons projected)
        shm = self._shadow_mask(x0, y0, w, h)
        shc = cc('#3a3480') * (np.clip(shade, 0, 2)[..., None] ** 1.4 * 0.95 + 0.07)
        shc = shc * (1 - 0.3 * graz[..., None]) + refl * 0.3 * graz[..., None] * cc('#b0a0d0')
        # painted variation INSIDE the shadow bands: broad tonal drift, glossy tar patches (catching the
        # cool sky), bleached aggregate chips and dark specks (world space, faded with footprint)
        tpf = cv2.remap(C.fbm(512, 512, 10, 4, seed=41), ((lat * 6.0) % 512).astype(np.float32),
                        ((cs * 4.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        tar = C.smoothstep(0.56, 0.61, tpf) * (1 - C.smoothstep(HALF - 0.5, HALF - 0.1, np.abs(lat)))
        self.tar = tar
        drift = 1 + 0.2 * (g2 - 0.5) + 0.1 * (g1 - 0.5) * fadeg
        shc = shc * drift[..., None] * (1 - 0.38 * tar[..., None])
        shc = shc + cc('#6a78c8') * (0.14 * tar * (0.4 + 0.6 * C.smoothstep(0.4, 0.7, fine_)))[..., None]
        agg_l = C.smoothstep(0.34, 0.27, gr1) * gfade * (1 - tar)
        agg_d = C.smoothstep(0.68, 0.76, gr1) * gfade
        shc = shc * (1 + 0.32 * agg_l - 0.22 * agg_d)[..., None]
        shc = shc + cc('#8a86c8') * (0.05 * C.smoothstep(0.55, 0.75, g1) * fadeg)[..., None]
        # ---- warm backlit sheen + aggregate glints on the lit asphalt between the rail shadows (the low sun
        # skims the road from ahead: strongest toward the sun column and along the sea-side lane)
        sunx = sc.sun_p[0]
        ks = np.exp(-((xs - sunx) / (0.32 * W)) ** 2)
        side = C.smoothstep(-1.0, HALF, lat)
        near_sh = np.clip(C.blur(shm, 4.0 * self.u) * 1.6, 0, 1) * (1 - shm)     # lit asphalt next to shadow
        sheen = (0.12 + 0.3 * near_sh) * side * ks * (1 - shm) * (0.4 + 0.6 * np.clip(1 - graz * 1.5, 0, 1))
        col = col + cc('#ffa860') * sheen[..., None] * shade[..., None]
        # broad warm light pool on the open (sun-side, mid-distance) road vs a cool shade on the inner lane
        # under the hill: a painted warm / cool split rather than uniform shading
        pool = np.exp(-((xs - sunx * 0.85) / (0.42 * W)) ** 2) * C.smoothstep(-2.5, 1.0, lat) * (1 - shm)
        pool = pool * C.smoothstep(sc.hzp + 0.02 * H, sc.hzp + 0.12 * H, ys)
        col = col * (1 + np.array([0.2, 0.07, -0.06], np.float32) * pool[..., None])
        inner = C.smoothstep(-0.8, -HALF + 0.2, lat)
        col = col * (1 - np.array([0.2, 0.15, -0.04], np.float32) * inner[..., None])
        # warm specular sheen band along the sun-facing far (sea-side) lane: the low sun skims the asphalt
        farl = C.smoothstep(0.2, 1.4, lat) * C.smoothstep(HALF - 0.1, HALF - 0.5, lat)
        dband = np.exp(-((dy - 0.13 * H) / (0.07 * H)) ** 2)
        col = col + cc('#ffb070') * (0.2 * farl * dband * ks * (1 - shm) * (0.7 + 0.3 * g2))[..., None]
        fine = cv2.remap(C.fbm(512, 512, 70, 2, seed=23), ((lat * 30.0) % 512).astype(np.float32),
                         ((cs * 30.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        spk = C.smoothstep(0.76, 0.86, fine) * np.clip(1 - fp * 30, 0, 1) * (1 - shm) * ks * side
        col = col + np.array([1.4, 0.95, 0.55], np.float32) * (spk * (0.25 + 0.6 * near_sh))[..., None]
        # the glossy tar in the cracks catches the low sun: broken warm specular threads toward the sun column
        brk = C.smoothstep(0.45, 0.7, fine)
        cspec = cmap2 * brk * (ks ** 2) * (1 - shm) * (0.3 + 0.7 * side)
        ks2 = np.exp(-((xs - sunx) / (0.5 * W)) ** 2)
        cspec = cmap2 * brk * ks2 * (1 - shm) * (0.4 + 0.6 * side)
        col = col + np.array([1.8, 1.15, 0.62], np.float32) * (cspec * 1.1)[..., None]
        col = col * (1 - 0.22 * tar[..., None]) + cc('#ffb070') * (0.12 * tar * ks * side)[..., None]
        col = col * (1 - shm[..., None]) + shc * shm[..., None]
        # ---- markings (worn white, lit warm / cool in shadow)
        e = band(lat, HALF - 0.45, 0.075, fpl) + band(lat, -(HALF - 0.45), 0.075, fpl)
        dash = band((cs % 10.0), 2.5, 2.5, fps)
        ctr = band(lat, 0.0, 0.075, fpl) * dash
        wear = np.clip(0.75 + 0.5 * (g1 - 0.5) * fadeg + 0.25, 0, 1)
        chip = cv2.remap(C.fbm(512, 512, 60, 3, seed=37), ((lat * 20.0) % 512).astype(np.float32),
                         ((cs * 6.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        chip = C.smoothstep(0.56, 0.6, chip) * np.clip(1.4 - fp * 30, 0, 1)
        mk = np.clip(e + ctr, 0, 1) * wear * (1 - 0.8 * chip) * (0.88 + 0.12 * g2)
        wl = cc('#fff0e0') * (1 - shm[..., None]) + cc('#b8b4e6') * shm[..., None]
        wl = wl * (1 - 0.35 * graz[..., None]) + refl * 0.35 * graz[..., None]
        col = col * (1 - mk[..., None] * 0.92) + wl * mk[..., None] * 0.92
        # the sea-side edge line catches the low sun: broken warm specular glints on the lit stretches
        eln = band(lat, HALF - 0.45, 0.075, fpl) * wear
        gb = C.smoothstep(0.55, 0.7, fine) * (1 - shm) * ks2
        col = col + np.array([2.0, 1.35, 0.75], np.float32) * (eln * gb * 0.9)[..., None]
        # wet-looking sheen along ALL painted lines where they face the backlight (smooth, not sparkly)
        wet = mk * (1 - shm) * ks2 * (0.35 + 0.65 * side) * C.smoothstep(0.35, 0.6, g2)
        col = col + np.array([0.9, 0.6, 0.35], np.float32) * (wet * 0.45)[..., None]
        # rim + glint on the manhole covers facing the sun
        col = col + np.array([1.2, 0.8, 0.45], np.float32) * (mring * ks2 * (1 - shm) * 0.35)[..., None]
        # ---- near lane: broad painted tonal breakup (sun-bleached / damp / resealed patches) with crisp edges
        nb_ = np.clip(1.5 - fp * 30, 0, 1)
        bl = cv2.remap(C.fbm(512, 512, 9, 4, seed=43), ((lat * 7.0) % 512).astype(np.float32),
                       ((cs * 4.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        bl2 = cv2.remap(C.fbm(512, 512, 14, 3, seed=44), ((lat * 9.0) % 512).astype(np.float32),
                        ((cs * 6.0) % 512).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        vb = (bl - 0.5) * 2 * nb_
        col = col * (1 + vb[..., None] * np.array([0.15, 0.12, 0.08], np.float32))
        damp = C.smoothstep(0.6, 0.625, bl2) * nb_
        col = col * (1 - damp[..., None] * np.array([0.12, 0.1, 0.04], np.float32))
        bleach = C.smoothstep(0.37, 0.345, bl2) * nb_ * (1 - C.smoothstep(HALF - 0.6, HALF - 0.2, np.abs(lat)))
        col = col * (1 + bleach[..., None] * np.array([0.1, 0.08, 0.06], np.float32))
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
        stripes = [(0.46, 0.52, '#6e6a9a'), (0.52, 0.6, '#9a94c0'), (0.6, 0.665, '#625e8e'),
                   (0.665, 0.74, '#aca6d0'), (0.74, 0.8, '#7e79a8')]
        for h0, h1, c in stripes:
            cv.zmode = ('Y', 0.5 * (h0 + h1))
            cv.poly(self.strip(0.3, s1, RAIL, h0, RAIL, h1, n=1400), cc(c))
        # wear: road grime splashed on the lower flute (patchy), rust bleeding from the post bolts,
        # a dull dent here and there
        wrng = np.random.default_rng(8)
        cv.zmode = ('Y', 0.5)
        for s0 in np.arange(0.3, min(s1, 120.0), 1.3):
            s1_ = s0 + wrng.uniform(0.6, 1.4)
            if self.zof(s0, RAIL) < 2.0:
                continue
            cv.poly(self.strip(s0, s1_, RAIL, 0.46, RAIL, 0.46 + wrng.uniform(0.04, 0.12), n=8), cc('#4a4058'),
                    alpha=wrng.uniform(0.15, 0.5))
        for sp in np.arange(2.0, min(s1, 90.0), 4.0):
            if self.zof(sp, RAIL) < 2.0:
                continue
            cv.zmode = ('Y', 0.6)
            for k in range(int(wrng.integers(1, 3))):
                o = wrng.uniform(-0.12, 0.12)
                ln = wrng.uniform(0.08, 0.2)
                cv.poly(self.strip(sp + o, sp + o + wrng.uniform(0.02, 0.05), RAIL, 0.65 - ln, RAIL, 0.66, n=4),
                        cc('#9a5a40'), alpha=wrng.uniform(0.35, 0.7))
            # bolt heads on the beam
            cv.poly(self.strip(sp - 0.03, sp + 0.03, RAIL, 0.625, RAIL, 0.655, n=4), cc('#4c4466'), alpha=0.9)
            if wrng.random() < 0.25:
                d0 = sp + wrng.uniform(0.8, 2.8)
                cv.zmode = ('Y', 0.7)
                cv.poly(self.strip(d0, d0 + wrng.uniform(0.3, 0.6), RAIL, 0.68, RAIL, 0.75, n=6), cc('#6c6590'),
                        alpha=0.45)
        # warm sheen on the convex upper flute where it faces the sun (broken by the dents / grime)
        sx_, sy_ = sc.sun_p
        ss_ = np.linspace(0.3, s1, 700)
        px_, py_ = self.P(ss_, RAIL, 0.7)
        k_ = 0.35 + 0.65 * np.exp(-((px_ - sx_) / (0.35 * sc.W)) ** 2)
        zz_ = self.zof(ss_, RAIL)
        for a in range(0, len(ss_) - 1, 5):
            b = min(a + 6, len(ss_))
            ka = float(k_[a:b].mean()) * wrng.uniform(0.75, 1.0)
            if ka < 0.03 or zz_[a] < 1.5:
                continue
            # the whole convex upper flute takes the warm light (a continuous band, brightest toward the sun)
            cv.zmode = ('Y', 0.72)
            cv.poly(self.strip(ss_[a], ss_[b - 1] + 0.02, RAIL, 0.665, RAIL, 0.79, n=4), cc('#f0b088'),
                    alpha=min(ka * 0.75, 0.8))
            cv.poly(self.strip(ss_[a], ss_[b - 1] + 0.02, RAIL, 0.72, RAIL, 0.775, n=4), cc('#ffcc98'),
                    alpha=min(ka * 0.7, 0.75))
        # hot top edge where the low sun grazes the rail
        cv.zmode = ('Y', 0.8)
        top = self.strip(0.3, s1, RAIL, 0.79, RAIL, 0.815, n=1400)
        cv.poly(top, cc('#dcb4ac'))
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
            cv.line(np.stack([px_[a:b], py_[a:b]], 1), wd, hot, alpha=min(ka * 0.5, 0.55))
        # cool sky reflected in the lower flute + a thin dark shadow line under the beam
        cv.zmode = ('Y', 0.49)
        cv.poly(self.strip(0.3, s1, RAIL, 0.46, RAIL, 0.53, n=1400), cc('#6a74cc'), alpha=0.75)
        cv.poly(self.strip(0.3, s1, RAIL, 0.44, RAIL, 0.465, n=1400), cc('#2c2848'), alpha=0.8)
        # hot glint streak on the sun-facing top edge, broken at every post (the beam joints)
        ss_ = np.linspace(0.3, s1, 2400)
        px_, py_ = self.P(ss_, RAIL, 0.808)
        zz_ = self.zof(ss_, RAIL)
        k_ = 0.42 + 0.25 * np.exp(-((px_ - sx_) / (0.6 * sc.W)) ** 2) + 0.55 * np.exp(-((px_ - sx_) / (0.25 * sc.W)) ** 2)
        gap = np.abs((ss_ % 4.0) - 2.0) < 0.03          # only a hairline break at the beam joints
        hot2 = np.array([1.6, 0.82, 0.36], np.float32)
        a = 0
        while a < len(ss_) - 1:
            b = a
            while b < len(ss_) - 1 and gap[b] == gap[a] and b - a < 8:
                b += 1
            if not gap[a] and zz_[a] > 1.2:
                ka = float(k_[a:b + 1].mean())
                if ka > 0.04:
                    cv.zmode = ('c', 1.0 / zz_[a:b + 1].mean())
                    wd = max(sc.f * 0.045 / zz_[a:b + 1].mean(), 1.4 * self.u)
                    cv.line(np.stack([px_[a:b + 1], py_[a:b + 1]], 1), wd, hot2, alpha=min(ka * 1.4, 1.0))
                    if ka > 0.35:
                        cv.line(np.stack([px_[a:b + 1], py_[a:b + 1] - wd * 0.15], 1), wd * 0.4,
                                np.array([2.4, 1.5, 0.7], np.float32), alpha=min((ka - 0.35) * 1.6, 1.0))
            a = b
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
        hd = self._hill_depth()
        cv.zmode = ('F', hd)
        self.tree.zmode = ('F', hd)
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
        # per-cell value / hue variation: some cells sun-dried (olive-ochre), some lush, some bare and dark;
        # each cell a touch lighter at its upper edge where the grass tips catch the sky
        crng = np.random.default_rng(58)
        tints = [cc('#5a5a2c'), cc('#3c5230'), cc('#1c2a26'), cc('#4a4a34'), cc('#2e4632')]
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            for u0 in np.arange(0.0, UT - 0.01, 2.0):
                X0, Y0, Z0 = self.lat_pt(np.array([sp, sp + 2.0, sp + 2.0, sp]), np.array([u0, u0, u0 + 2.0, u0 + 2.0]))
                if Z0.min() < 1.5:
                    continue
                qx, qy = sc.proj(X0, Y0, Z0)
                c_ = tints[int(crng.integers(0, len(tints)))]
                self._clip_put(cv, cv.poly_mask(list(zip(qx, qy))), c_, hm, float(crng.uniform(0.3, 0.7)))
                X1, Y1, Z1 = self.lat_pt(np.array([sp, sp + 2.0, sp + 2.0, sp]),
                                         np.array([u0 + 1.4, u0 + 1.4, u0 + 2.0, u0 + 2.0]))
                qx, qy = sc.proj(X1, Y1, Z1)
                self._clip_put(cv, cv.poly_mask(list(zip(qx, qy))), cc('#6a6a48'), hm, float(crng.uniform(0.08, 0.2)))
        self._cell_grass(cv, hm, UT)
        # cells darken toward the foot of the slope (weed occlusion) BEFORE the frame: the concrete stays clean
        self._cell_texture(cv, hm, UT)
        # ---- cast-in-place concrete frame: every beam is a solid 3-D bar standing HB out of the slope, projected
        # with the camera, so its faces converge toward the road's vanishing point: a lit front face, a warm
        # top face on the crossbars (facing up the slope / the sky), and the shadowed near side face of the
        # up-slope beams (the sun is ahead: the faces turned toward the camera are in shade)
        bw = 0.35
        HB = 0.24
        front_c = cc('#9c8aa6')
        side_c = cc('#40385a')
        top_c = cc('#c09c98')
        bm = np.zeros((cv.H, cv.W), np.float32)
        topm = np.zeros((cv.H, cv.W), np.float32)
        hbm = np.zeros((cv.H, cv.W), np.float32)
        nd_, nY_ = 1.0 / math.hypot(1.0, 0.55), 0.55 / math.hypot(1.0, 0.55)   # outward slope normal (d, Y)

        def P3(s_, u_, h_):
            X_, Z_, nx_, nz_ = self.at(s_)
            d_ = -(HALF + 0.6) - np.asarray(u_) * 0.55 + nd_ * h_
            Y_ = np.asarray(u_) + nY_ * h_
            return sc.proj(X_ + nx_ * d_, Y_, Z_ + nz_ * d_), Z_ + nz_ * d_

        def quad(a, b):
            (ax_, ay_), (bx_, by_) = a, b
            return list(zip(ax_, ay_)) + list(zip(bx_[::-1], by_[::-1]))

        def acc(res, dst):
            if res is None:
                return res
            m_, x_, y_ = res
            sl_ = dst[y_:y_ + m_.shape[0], x_:x_ + m_.shape[1]]
            np.maximum(sl_, m_, out=sl_)
            return res
        # soft contact / cast shadow of the frame on the cells (the low sun ahead throws it toward the camera)
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            uu = np.linspace(0, UT + bw, 30)
            (a_, za) = P3(np.full_like(uu, sp - 0.35), uu, 0.0)
            (b_, zb) = P3(np.full_like(uu, sp), uu, 0.0)
            if min(za.min(), zb.min()) < 1.5:
                continue
            self._clip_put(cv, cv.poly_mask(quad(a_, b_)), cc('#101a22'), hm, 0.45)
        # up-slope beams: near side face (shade) then the front face
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            uu = np.linspace(0, UT + bw, 30)
            (a0, za) = P3(np.full_like(uu, sp), uu, 0.0)
            (a1, _z) = P3(np.full_like(uu, sp), uu, HB)
            (b1, zb) = P3(np.full_like(uu, sp + bw), uu, HB)
            if min(za.min(), zb.min()) < 1.5:
                continue
            self._clip_put(cv, acc(cv.poly_mask(quad(a0, a1)), bm), side_c, hm)
            self._clip_put(cv, acc(cv.poly_mask(quad(a1, b1)), bm), front_c, hm)
        # crossbars along the slope: cast shadow below, top face, front face (drawn over the up-slope beams so
        # the joints read as one poured frame)
        for u in np.arange(0.0, UT + 0.01, 2.0):
            (f0, z0) = P3(ss, u, HB)
            (f1, _z) = P3(ss, u + bw, HB)
            (t1, _z) = P3(ss, u + bw, 0.0)
            (c0, _z) = P3(ss, u - 0.3, 0.0)
            (g0, _z) = P3(ss, u, 0.0)
            ok = z0 > 1.5
            sel = lambda q: (q[0][ok], q[1][ok])
            self._clip_put(cv, cv.poly_mask(quad(sel(g0), sel(c0))), cc('#141c26'), hm, 0.55)
            self._clip_put(cv, acc(acc(cv.poly_mask(quad(sel(f1), sel(t1))), bm), topm), top_c, hm)
            self._clip_put(cv, acc(acc(cv.poly_mask(quad(sel(f0), sel(f1))), bm), hbm), front_c, hm)
            # dark lower arris of the crossbar (it hangs over the grass)
            (e0, _z) = P3(ss, u + 0.05, HB)
            self._clip_put(cv, cv.poly_mask(quad(sel(f0), sel(e0))), cc('#4a4062'), hm, 0.8)
        self.hbm = hbm
        self._lattice_finish(cv, bm, topm, r, hm)
        self._lattice_sun(cv, bm, r, hm)
        # backlit grass tufts in the cells, over the frame: dark blades whose sun-side edges catch a warm rim
        trng = np.random.default_rng(615)
        tcv = P.Canvas(cv.W, cv.H)
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            for u0 in np.arange(0.0, UT - 0.01, 2.0):
                for _k in range(int(trng.integers(3, 7))):
                    X_, Y_, Z_ = self.lat_pt(sp + trng.uniform(0.45, 1.9), u0 + bw + trng.uniform(0.05, 0.9))
                    if Z_ < 2.2:
                        continue
                    bx_, by_ = sc.proj(X_, Y_, Z_)
                    hh_ = sc.f * trng.uniform(0.3, 0.6) / Z_
                    if hh_ < 5:
                        continue
                    P.grass_tuft(tcv, bx_, by_, hh_, trng, cc('#1c2a26') * trng.uniform(0.9, 1.4), cc('#ffb070'),
                                 lean=0.22, n=int(trng.integers(5, 10)), width=0.7, rim_amt=0.95)
        M_, hx_, hy_ = hm
        k_ = np.zeros((cv.H, cv.W), np.float32)
        k_[hy_:hy_ + M_.shape[0], hx_:hx_ + M_.shape[1]] = M_
        ta = np.clip(tcv.a, 0, 1) * k_
        cv.rgb = cv.rgb * (1 - ta[..., None]) + tcv.rgb * k_[..., None]
        cv.a = ta + cv.a * (1 - ta)
        del tcv
        # gutter at the road edge
        g = self.strip(0.3, self.SV + 5, -HALF - 0.6, 0, -HALF, 0, n=300)
        self._clip_put(cv, cv.poly_mask(g), cc('#3a3050'), hm)
        g = self.strip(0.3, self.SV + 5, -HALF - 0.12, 0, -HALF, 0, n=300)
        self._clip_put(cv, cv.poly_mask(g), cc('#a08aa8'), hm, 0.8)

    def _lattice_sun(self, cv, bm, face, hm):
        """low evening sun raking across the slope through the crown: crisp-edged warm dapples on the beams
        (hot top lips) and dry-grass highlights in the cells, the rest of the face cooler; the upper slope
        under the crown stays in its shadow."""
        sc = self.sc
        u = self.u
        fm, fx0, fy0 = face
        M, hx, hy = hm
        h, w = fm.shape
        F = fm.astype(np.float32).copy()
        clipf = np.zeros((cv.H, cv.W), np.float32)
        clipf[hy:hy + M.shape[0], hx:hx + M.shape[1]] = M
        F *= clipf[fy0:fy0 + h, fx0:fx0 + w]
        B = bm[fy0:fy0 + h, fx0:fx0 + w] * F
        rgb = cv.rgb[fy0:fy0 + h, fx0:fx0 + w]
        # dapple pattern: two octaves, stretched along the sun direction (long light slivers)
        n1 = cv2.resize(C.fbm(max(w // 3, 8), max(h // 5, 8), max(w / (70 * u) / 3, 3), 4, seed=661, aspect=False),
                        (w, h), interpolation=cv2.INTER_CUBIC)
        n2 = C.fbm(w, h, max(w / (24 * u), 3), 3, seed=662)
        nn = 0.7 * n1 + 0.3 * n2
        # the crown shades the upper slope and the left: light grows toward the road / bend and the foot
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, w, dtype=np.float32)[None]
        open_ = np.clip(0.2 + 0.55 * yy + 0.45 * xx, 0, 1)
        lit = C.smoothstep(0.6 - 0.25 * open_, 0.63 - 0.25 * open_, nn) * F
        lit = cv2.GaussianBlur(lit, (0, 0), 0.7 * u)
        # beams: warm lit faces, hot top lips
        warm = np.array([0.5, 0.26, 0.02], np.float32)
        rgb *= (1 + (lit * B)[..., None] * warm)
        # cells: dry-grass highlights (streaky, only where the sun lands)
        gr = rng_ = np.random.default_rng(663)
        st = cv2.GaussianBlur(rng_.standard_normal((h, w)).astype(np.float32), (0, 0), sigmaX=0.8 * u, sigmaY=4.5 * u)
        st = st / (st.std() + 1e-6)
        blades = C.smoothstep(0.3, 1.3, st)
        cell = (1 - B) * F
        dry = cc('#c8a060')
        rgb[:] = rgb + (dry - rgb) * (0.55 * lit * cell * blades)[..., None]
        rgb[:] = rgb * (1 + (0.22 * lit * cell * (1 - blades))[..., None] * np.array([1.0, 0.8, 0.3], np.float32))
        # the unlit face: a touch cooler / deeper (warm / cool split)
        sh = (1 - lit) * F
        rgb *= (1 - sh[..., None] * np.array([0.1, 0.07, 0.0], np.float32))
        cv.rgb[fy0:fy0 + h, fx0:fx0 + w] = rgb

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
        d = 10.0 * u
        Msh = np.float32([[1, 0, -d], [0, 1, d * 0.55]])
        sh = cv2.warpAffine(B, Msh, (w, h))
        sh = cv2.GaussianBlur(sh, (0, 0), 1.6 * u) * (1 - B) * F
        rgb *= (1 - 0.58 * sh)[..., None]
        # --- concrete: mottling, vertical water stains, chips; darker toward the foot of the slope
        n1 = C.fbm(w, h, max(w / (60 * u), 3), 4, seed=91)
        st = cv2.resize(C.fbm(max(w // 3, 8), max(h // 24, 4), max(w / (14 * u) / 3, 3), 3, seed=92, aspect=False),
                        (w, h), interpolation=cv2.INTER_CUBIC)
        stain = C.smoothstep(0.52, 0.78, st)
        rng = np.random.default_rng(93)
        chips = (rng.random((h, w)) > 0.994).astype(np.float32)
        chips = cv2.GaussianBlur(chips, (0, 0), 0.8 * u) * 3.0
        tex = (0.78 + 0.46 * n1) * (1 - 0.42 * stain) * (1 - 0.4 * np.clip(chips, 0, 1))
        # vertical rain streaks running down from every beam (long, thin, darker) + pale lime bloom
        rs = rng.standard_normal((h, w)).astype(np.float32)
        rs = cv2.GaussianBlur(rs, (0, 0), sigmaX=1.1 * u, sigmaY=16 * u)
        rs = rs / (rs.std() + 1e-6)
        n3 = C.fbm(w, h, max(w / (90 * u), 3), 3, seed=95)
        tex *= 1 - 0.22 * C.smoothstep(0.6, 1.8, rs) * C.smoothstep(0.35, 0.65, n3)
        tex *= 1 + 0.14 * C.smoothstep(1.0, 2.2, -rs) * C.smoothstep(0.55, 0.75, n1)
        # aggregate speckle (fine, static) and chipped arrises: dark notches along the beam edges
        sp_ = rng.standard_normal((h, w)).astype(np.float32)
        tex *= 1 + 0.05 * np.clip(cv2.GaussianBlur(sp_, (0, 0), 0.6 * u), -2, 2)
        er = np.clip(B - cv2.erode(B, np.ones((3, 3), np.uint8)), 0, 1)
        notch = C.smoothstep(0.62, 0.72, C.fbm(w, h, max(w / (8 * u), 4), 2, seed=96))
        tex *= 1 - 0.4 * er * notch
        # lichen / moss blotches, mostly on the lower beams
        moss = C.smoothstep(0.6, 0.72, C.fbm(w, h, max(w / (25 * u), 4), 3, seed=97)) * \
            np.clip((np.arange(h, dtype=np.float32)[:, None] / max(h, 1)) * 1.4 - 0.2, 0, 1)
        rgb[:] = rgb + (C.hex2rgb('#4a5a3a') * 0.9 - rgb) * (moss * B * 0.45)[..., None]
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
        # --- water stains streaked down from every crossbar: a dark wash that starts under the bar and
        # fades down the slope beams in thin vertical runs (plus pale lime-scale runs), on the concrete only
        HB = self.hbm[y0:y1, x0:x1] * clipf[y0:y1, x0:x1]
        dn = HB.copy()
        stp = max(int(round(1.5 * u)), 1)
        for _ in range(int(45 * u / stp)):
            dn = np.maximum(dn, np.pad(dn, ((stp, 0), (0, 0)))[:h] * 0.965)
        below = np.clip(dn - HB, 0, 1)
        runs = rng.standard_normal((h, w)).astype(np.float32)
        runs = cv2.GaussianBlur(runs, (0, 0), sigmaX=1.3 * u, sigmaY=22 * u)
        runs = runs / (runs.std() + 1e-6)
        wet = below * C.smoothstep(-0.2, 1.2, runs) * B
        rgb *= (1 - 0.5 * wet)[..., None]
        rgb[:] = rgb + (cc('#2c3440') * 0.8 - rgb) * (wet * 0.25)[..., None]
        lime = below * C.smoothstep(1.3, 2.3, -runs) * B * 0.5
        rgb[:] = rgb + (cc('#b8b0a4') - rgb) * lime[..., None] * 0.5
        # rust-brown bleed right under each bar
        und2 = np.clip(np.pad(HB, ((int(3 * u) + 1, 0), (0, 0)))[:h] - HB, 0, 1) * B
        rgb[:] = rgb + (cc('#4a3430') - rgb) * (und2 * 0.5)[..., None]
        # --- chipped arrises: pale exposed aggregate flakes with a dark notch below, along the beam edges
        er2 = np.clip(B - cv2.erode(B, np.ones((5, 5), np.uint8)), 0, 1)
        chipn = C.fbm(w, h, max(w / (5 * u), 4), 2, seed=98)
        chip = er2 * C.smoothstep(0.66, 0.72, chipn)
        rgb[:] = rgb + (cc('#c8bcb0') * 0.85 - rgb) * (chip * 0.7)[..., None]
        chipd = np.clip(np.pad(chip, ((max(int(1.5 * u), 1), 0), (0, 0)))[:h] - chip, 0, 1)
        rgb *= (1 - 0.5 * chipd)[..., None]
        # --- warm sun glint along the top edges of the lit crossbars (broken, hot where the edge faces the sun)
        TM = topm[y0:y1, x0:x1] * clipf[y0:y1, x0:x1]
        TMe = np.clip(TM - np.pad(TM, ((max(int(1.2 * u), 1), 0), (0, 0)))[:h], 0, 1)
        gn = C.fbm(w, h, max(w / (30 * u), 4), 2, seed=99)
        xsr = np.arange(w, dtype=np.float32)[None] / max(w, 1)
        gl_ = TMe * C.smoothstep(0.42, 0.6, gn) * (0.55 + 0.45 * xsr)
        rgb[:] = rgb + (np.array([1.9, 1.2, 0.7], np.float32) - rgb) * np.clip(gl_ * 1.1, 0, 1)[..., None]
        # --- weathered grain running ALONG each beam (streaks follow the local beam direction), broad tonal
        # breakup from segment to segment, and lost (soft) edges on the shade side (away from the sun)
        Bb = cv2.GaussianBlur(B, (0, 0), 2.5 * u)
        gyb, gxb = np.gradient(Bb)
        J11 = cv2.GaussianBlur(gxb * gxb, (0, 0), 7 * u)
        J22 = cv2.GaussianBlur(gyb * gyb, (0, 0), 7 * u)
        J12 = cv2.GaussianBlur(gxb * gyb, (0, 0), 7 * u)
        th = 0.5 * np.arctan2(2 * J12, J11 - J22)                  # across-beam direction
        cth, sth = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
        xxg = np.arange(w, dtype=np.float32)[None] + x0
        yyg = np.arange(h, dtype=np.float32)[:, None] + y0
        ua = xxg * cth + yyg * sth
        va = -xxg * sth + yyg * cth
        gt = C.fbm(512, 512, 96, 3, seed=191)
        grain = cv2.remap(gt, ((ua / (1.3 * u)) % 512).astype(np.float32), ((va / (26 * u)) % 512).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        bt = C.fbm(512, 512, 8, 3, seed=192)
        blot = cv2.remap(bt, ((ua / (5 * u)) % 512).astype(np.float32), ((va / (9 * u)) % 512).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        gv = (grain - 0.5) * 2
        bv = (blot - 0.5) * 2
        wb = B[..., None]
        rgb[:] = rgb * (1 + wb * (0.17 * gv + 0.2 * bv)[..., None] * np.array([1.0, 0.94, 0.85], np.float32))
        # a few long dark checks / splits along the grain
        split = C.smoothstep(0.8, 0.9, grain) * C.smoothstep(0.2, 0.5, blot)
        rgb *= (1 - 0.35 * split * B)[..., None]
        # shade-side edges (outward normal turned away from the sun, i.e. left / down) go soft
        gmb = np.hypot(gxb, gyb) + 1e-6
        outn = (-gxb * 0.95 + gyb * 0.3) / gmb                    # outward normal . (-sun dir)
        k5 = np.ones((5, 5), np.uint8)
        band_ = np.clip(cv2.dilate(B, k5) - cv2.erode(B, k5), 0, 1)
        lostw = band_ * C.smoothstep(0.1, 0.6, outn)
        soft = cv2.GaussianBlur(rgb, (0, 0), 1.8 * u)
        rgb[:] = rgb + (soft - rgb) * (0.85 * lostw)[..., None]
        cv.rgb[y0:y1, x0:x1] = rgb
        # --- drip streaks: dark grime running down the slope from the underside of every horizontal beam,
        # and moss cushions packed into the lattice joints (green-dark body, a warm-lit top)
        drng = np.random.default_rng(97)
        grime = cc('#26262e')
        for ub in (2.0, 4.0):
            for sp in np.arange(0.4, self.SV + 5, 0.35):
                if drng.random() < 0.45:
                    continue
                s_ = sp + drng.uniform(-0.15, 0.15)
                ln = drng.uniform(0.25, 1.1)
                X0_, Y0_, Z0_ = self.lat_pt(s_, ub)
                X1_, Y1_, Z1_ = self.lat_pt(s_ + drng.uniform(-0.03, 0.03), ub - ln)
                if min(Z0_, Z1_) < 2.0:
                    continue
                a_ = sc.proj(X0_, Y0_, Z0_); b_ = sc.proj(X1_, Y1_, Z1_)
                wd = max(sc.f * drng.uniform(0.04, 0.1) / Z0_, 0.6)
                tt = np.linspace(0, 1, 6)
                pts = [(a_[0] + (b_[0] - a_[0]) * t_, a_[1] + (b_[1] - a_[1]) * t_) for t_ in tt]
                for k in range(5):
                    self._clip_put(cv, cv.line_mask(pts[k:k + 2], wd * (1 - 0.15 * k)), grime, hm,
                                   0.6 * (1 - k / 5.0) ** 0.8)
        moss_d, moss_l = cc('#1c2a22'), cc('#56603a')
        for ub in (0.0, 2.0, 4.0):
            for sp in np.arange(0.5, self.SV + 5, 2.0):
                X_, Y_, Z_ = self.lat_pt(sp + 0.3, ub + 0.17)
                if Z_ < 2.5:
                    continue
                cx_, cy_ = sc.proj(X_, Y_, Z_)
                R_ = sc.f * drng.uniform(0.11, 0.2) / Z_
                if R_ < 1.2:
                    continue
                for k in range(int(drng.integers(3, 6))):
                    ox_, oy_ = drng.normal(0, 0.45 * R_), drng.normal(0.25 * R_, 0.3 * R_)
                    rr = R_ * drng.uniform(0.35, 0.6)
                    th = np.linspace(0, 2 * np.pi, 14)
                    rj = rr * (1 + 0.18 * np.sin(th * 3 + drng.uniform(0, 6)))
                    poly = list(zip(cx_ + ox_ + rj * np.cos(th), cy_ + oy_ + rj * 0.7 * np.sin(th)))
                    self._clip_put(cv, cv.poly_mask(poly), moss_d, hm, 0.7)
                    top = list(zip(cx_ + ox_ + 0.2 * rr + rj * 0.7 * np.cos(th), cy_ + oy_ - 0.25 * rr + rj * 0.35 * np.sin(th)))
                    self._clip_put(cv, cv.poly_mask(top), moss_l, hm, 0.45)
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
        k = max(int(0.03 * W), 1)
        clipm = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        clipm = C.smoothstep(0.3, 0.7, cv2.GaussianBlur(clipm, (0, 0), 0.01 * W))
        clip = (clipm, hx, hy)
        fill, ridge = [], []
        n = len(rid)
        # skyline crowns: distinct cauliflower heads spaced along the ridge (a big one, then smaller
        # ones), pushed out across the ridge line by varying amounts so the outline steps
        i = 0.0
        tan_ = np.gradient(rid, axis=0)
        while i < n - 1:
            ii = int(i)
            x, y = rid[ii]
            f = ii / n
            r = (0.09 * (1 - f) ** 1.2 + 0.012) * W * rng.uniform(0.6, 1.5)
            tx_, ty_ = tan_[ii] / (np.hypot(*tan_[ii]) + 1e-9)
            nx_, ny_ = ty_, -tx_            # outward (up / right of the ridge running down-right)
            if ny_ > 0:
                nx_, ny_ = -nx_, -ny_
            o = r * rng.uniform(-0.15, 0.3)
            ridge.append((x + nx_ * o - r * 0.1, y + ny_ * o + r * 0.1, r, f))
            # step along the ridge by ~0.8 r (in samples)
            seg = np.hypot(*np.diff(rid[ii:ii + 2], axis=0)[0]) + 1e-6
            i += max(0.8 * r / seg * rng.uniform(0.8, 1.2), 1.0)
        for kk in range(14):
            j = rng.integers(0, len(tx))
            x, y = tx[j], ty[j]
            f = j / len(tx)
            up = rng.uniform(0, 1) ** 0.7
            x = x - up * rng.uniform(0.0, 0.25) * W * (1 - f)
            y = y - up * rng.uniform(0.0, 0.5) * H * (1 - f) ** 1.2
            r = (0.085 * (1 - f) ** 1.3 + 0.016) * W * rng.uniform(0.75, 1.25)
            y = min(y, ty[j] - 0.7 * r)          # clump bottoms stay clear of the lattice: branches show below
            fill.append((x, y, r, f))
        # two filler masses against the left frame edge (no gaps through to the dark hill base)
        fill += [(sc.ox + 0.004 * W, sc.oy + 0.3 * H, 0.05 * W, 0.0), (sc.ox + 0.01 * W, sc.oy + 0.47 * H, 0.045 * W, 0.0)]
        # clump hierarchy (s06_seaside_tree2): big cauliflower clumps, back (high / far) to front (low / near)
        cl = [l + (1,) for l in fill] + [l + (0,) for l in ridge]
        # skirt of smaller clumps hanging over the top of the lattice: a scalloped, overlapping crown foot
        # (no smooth horizontal bottom), with gaps where the trunks show
        srng = np.random.default_rng(41)
        gaps = [sc.ox + 0.05 * W, sc.ox + 0.12 * W]
        j = int(np.searchsorted(tx, sc.ox - 0.06 * W))
        while j < len(tx):
            x, y = tx[j], ty[j]
            if x > sc.ox + 0.17 * W:
                break
            r = (0.016 + 0.026 * srng.random() ** 1.5) * W
            if min(abs(x - g) for g in gaps) > 0.014 * W:
                cl.append((x + srng.normal(0, 4 * u), y + r * srng.uniform(-0.2, 0.55), r, 0.0, 0))
            seg = np.hypot(*(np.array([tx[min(j + 1, len(tx) - 1)] - x, ty[min(j + 1, len(ty) - 1)] - y]))) + 1e-6
            j += max(int(r * srng.uniform(0.7, 1.0) / seg), 1)
        cl.sort(key=lambda c: c[1] + 0.35 * c[2])
        # a few dark branches read in the gaps under the lowest clumps (above the lattice)
        brs = []
        brng = np.random.default_rng(31)
        for fx0 in (0.015, 0.05, 0.085, 0.12):
            j = int(np.argmin(np.abs(tx - sc.ox - fx0 * W)))
            bx0, by0 = tx[j], ty[j] + 6 * u
            pts = [(bx0, by0)]
            ang = -math.pi / 2 + brng.uniform(-0.45, 0.25)
            for k in range(7):
                ang += brng.normal(0, 0.25)
                ln = brng.uniform(18, 30) * u
                pts.append((pts[-1][0] + math.cos(ang) * ln, pts[-1][1] + math.sin(ang) * ln))
            brs.append((pts, brng.uniform(7, 11) * u))
            pts2 = [pts[2]]
            a2 = ang + brng.choice([-1, 1]) * 0.8
            for k in range(3):
                pts2.append((pts2[-1][0] + math.cos(a2) * 20 * u, pts2[-1][1] + math.sin(a2) * 20 * u))
            brs.append((pts2, 4 * u))
        # dark undergrowth along the top of the lattice (behind the trunks / branches)
        ug = []
        urng = np.random.default_rng(37)
        for j in range(0, int(len(tx) * 0.6), 6):
            f = j / len(tx)
            r = (0.03 * (1 - f) + 0.006) * W * urng.uniform(0.7, 1.3)
            ug.append((tx[j] + urng.normal(0, 4 * u), ty[j] - r * urng.uniform(0.2, 0.6), r, f * 0.6, 1))
        ug.sort(key=lambda c: c[1])
        ug_pal = dict(T2.PAL2_DARK, deep=cc('#122030'), shd=cc('#18293a'), sky=cc('#22394a'), half=cc('#30443f'),
                      lit=cc('#5e6436'), hot=cc('#b89048'))
        T2.canopy2(cv, ug, np.random.default_rng(38), unit=u * 0.8, clip=clip, pal=ug_pal, lit_bias=0.12,
                   leaf=0.8, sky_amt=0.8, soft_down=0.2, hang=0.2, far_haze=0.3, rim_amt=1.3)
        # contact shadow of the crown on the top of the lattice (the foliage sits ON the slope)
        for (u0, a_) in ((3.9, 0.2), (4.45, 0.35)):
            sm_ = ss[tx < sc.ox + 0.3 * W]
            if len(sm_) > 2:
                ax_, ay_ = sc.proj(*self.lat_pt(sm_, 5.2))
                bx_, by_ = sc.proj(*self.lat_pt(sm_, u0))
                self._clip_put(cv, cv.poly_mask(list(zip(ax_, ay_)) + list(zip(bx_[::-1], by_[::-1]))),
                               cc('#0e1024'), hm, a_)
        # round 13 painter (s06_seaside_tree10): two broad value masses (warm backlit / cool shade) cut by one
        # crisp dab-broken terminator, leaf-dab scallop silhouettes, lost shade-side edges, limbs behind
        # round 14 painter (s06_seaside_tree11): value read off the whole crown's shape against the sun at the
        # right (lit crown / merged shade body / navy underside), laid on as clustered leaf dabs, cauliflower
        # lobe silhouettes, hot rim only on sun-facing edges, lost edges on the shade side
        T11.canopy11(self.tree, cl, np.random.default_rng(29), unit=u, clip=clip, branches=brs,
                     ss=2, wcv=self.tw, anchor=(sc.ox, sc.oy, W, H))

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
                lob = sorted([(bx + rng.uniform(-0.7, 0.7) * r, by - r * rng.uniform(0.3, 0.6), r * rng.uniform(0.4, 0.6),
                               0.0, 0) for _ in range(int(rng.integers(2, 4)))], key=lambda c: c[1])
                T2.canopy2(cv, lob, rng, unit=self.u * 0.6, pal=T2.PAL2_DARK, far_haze=0.0, lit_bias=-0.1,
                           leaf=0.7, sky_amt=0.5, L=(-0.3, -0.5, -0.8))

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
                kk = 0.9 * np.exp(-(dd / 0.2) ** 2)          # warm rim only where the span crosses the sun glow
                self.wires.append((px.astype(np.float64), py.astype(np.float64), zz.astype(np.float64),
                                   np.clip(kk, 0, 1)))
                j = int(np.argmin(dd))
                if dd[j] < 0.4 and 0 < j < len(px) - 1:
                    self.wire_glints.append((px[j], py[j], 1.0 / zz[j], float(np.exp(-(dd[j] / 0.25) ** 2)),
                                             float(dx * 3 + Y)))

    def _cell_grass(self, cv, hm, UT):
        """dab-painted grass in the lattice cells: short blade strokes laid in clusters (a few values: dark
        roots, mid green, lit olive tips, warm sun-caught tips toward the sun side), denser at the top of each
        cell where the grass spills over the crossbar. Painted once into the plate: static, no shimmer."""
        sc = self.sc
        u = self.u
        rng = np.random.default_rng(612)
        M, hx, hy = hm
        lay = np.zeros((cv.H, cv.W, 3), np.float32)
        al = np.zeros((cv.H, cv.W), np.float32)
        cols = [cc('#1e2c28'), cc('#34482e'), cc('#56663a'), cc('#8a8446'), cc('#c89a5a')]
        sunx = sc.sun_p[0]
        for sp in np.arange(0.5, self.SV + 5, 2.0):
            for u0 in np.arange(0.0, UT - 0.01, 2.0):
                X0, Y0, Z0 = self.lat_pt(sp + 1.0, u0 + 1.0)
                if Z0 < 2.0:
                    continue
                scale = sc.f / Z0                  # px per metre
                if scale < 3:
                    continue
                n = int(np.clip(scale * scale * 0.9, 12, 420))
                ss_ = rng.uniform(0.35, 1.95, n)
                uu_ = 0.3 + 1.7 * rng.random(n) ** 0.6
                Xs, Ys, Zs = self.lat_pt(sp + ss_, u0 + uu_)
                px, py = sc.proj(Xs, Ys, Zs)
                ks = float(np.exp(-((float(np.mean(px)) - sunx) / (0.6 * sc.W)) ** 2))
                order = np.argsort(uu_)[::-1]            # far (upper) blades first
                for k in order:
                    hgt = scale * rng.uniform(0.12, 0.3)
                    lean = rng.normal(0.25, 0.25)
                    lit = uu_[k] / 2.0 * 0.6 + rng.uniform(0, 0.5)
                    ci = int(np.clip(lit * 3.2, 0, 3))
                    c_ = cols[ci]
                    if ci >= 2 and rng.random() < 0.35 * ks:
                        c_ = cols[4]
                    x0, y0 = px[k], py[k]
                    nb = int(rng.integers(2, 5))
                    for j in range(nb):
                        dx = (lean + rng.normal(0, 0.25)) * hgt * 0.5
                        th = max(int(round(scale * 0.02)), 1)
                        p0 = (int((x0 + j * 0.6 * u) * 4), int(y0 * 4))
                        p1 = (int((x0 + j * 0.6 * u + dx) * 4), int((y0 - hgt * rng.uniform(0.7, 1.0)) * 4))
                        cv2.line(lay, p0, p1, tuple(float(v) for v in c_), th, cv2.LINE_AA, 2)
                        cv2.line(al, p0, p1, 1.0, th, cv2.LINE_AA, 2)
        clip = np.zeros((cv.H, cv.W), np.float32)
        clip[hy:hy + M.shape[0], hx:hx + M.shape[1]] = M
        k = clip * 0.9
        a = np.clip(al, 0, 1) * k
        cv.rgb = cv.rgb * (1 - a[..., None]) + lay * k[..., None]
        cv.a = a + cv.a * (1 - a)

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
            for uu in (0.25, 2.25):
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
                r = ph * 0.6
                lob = sorted([(bx + rng.uniform(-0.9, 0.9) * r, by - r * rng.uniform(0.15, 0.5), r * rng.uniform(0.35, 0.6),
                               0.0, 0) for _ in range(int(rng.integers(2, 4)))], key=lambda c: c[1])
                T2.canopy2(cv, lob, rng, unit=self.u * 0.7, pal=T2.PAL2_DARK, far_haze=0.0, lit_bias=-0.08,
                           leaf=0.8, sky_amt=0.6)
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
