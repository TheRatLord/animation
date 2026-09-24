"""s05_sakura round 3: painted 2D cherry trees (one card per tree, placed at the trunk depth).

Each tree is painted like a background artist would:
* a trunk that forks into 3-5 tapering primary limbs (plus secondaries and drooping outer branches) that run up
  into the undersides of the blossom masses; warm dark brown-grey bark, sunlit warm rim on the sun side, cool sky
  bounce on the shadow side, vertical bark fissures;
* 3-5 big blossom masses per tree, each ONE smooth painted shape: large pink-white lit top, soft mid band,
  one continuous cool-lavender underside, warm peach transmitted light in the thin backlit edges;
  crisp edges + a cast shadow only where one mass overlaps another;
* the silhouette and the lower hem broken into lacy clusters of 5-petal florets with real gaps,
  10-20 sky pinholes through the thin parts, small hem clumps that hide limb segments.
"""
import math
import numpy as np
import cv2
from numba import njit

# palette (linear-ish display values)
LIT = np.array([0.98, 0.845, 0.9], np.float32)       # ~#FBE9F0
HOT = np.array([1.0, 0.9, 0.925], np.float32)
MID = np.array([0.96, 0.7, 0.83], np.float32)
LAV = np.array([0.72, 0.61, 0.8], np.float32)          # ~#B79AC8
DEEP = np.array([0.6, 0.5, 0.72], np.float32)
TRANS = np.array([1.0, 0.8, 0.76], np.float32)         # warm transmitted light
RIM = np.array([1.06, 0.97, 0.9], np.float32)


# ============================================================================ helpers

def _noise(h, w, cell, rng):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(n[:h, :w])


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _blur(a, sig):
    """Gaussian blur; large sigmas are done on a downsampled copy (fast)."""
    if sig <= 6:
        return cv2.GaussianBlur(a, (0, 0), sig)
    f = sig / 3.0
    h, w = a.shape[:2]
    sm = cv2.resize(a, (max(2, int(w / f)), max(2, int(h / f))), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 3.0)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _shift(a, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ============================================================================ numba rasterizers

@njit(cache=True, fastmath=True)
def limb_raster(segs, q, nx, ny, sv, lv, rv, av):
    """segs (n, 7): x0, y0, r0, x1, y1, r1, l0 (card px).  Keeps, per pixel, the segment whose axis is
    relatively closest (smooth unions at forks).  Writes the 2D normal (nx, ny scaled by d/r), the signed
    across-coordinate sv, the along-length lv, the local radius rv and AA coverage av."""
    H, W = q.shape
    for i in range(segs.shape[0]):
        x0, y0, r0, x1, y1, r1, l0 = segs[i, 0], segs[i, 1], segs[i, 2], segs[i, 3], segs[i, 4], segs[i, 5], segs[i, 6]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-9
        L = math.sqrt(L2)
        rm = max(r0, r1) + 2.0
        bx0 = max(0, int(min(x0, x1) - rm))
        bx1 = min(W, int(max(x0, x1) + rm) + 1)
        by0 = max(0, int(min(y0, y1) - rm))
        by1 = min(H, int(max(y0, y1) + rm) + 1)
        for y in range(by0, by1):
            for x in range(bx0, bx1):
                px = x + 0.5 - x0
                py = y + 0.5 - y0
                t = (px * dx + py * dy) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                rr = r0 + (r1 - r0) * t
                cx = px - t * dx
                cy = py - t * dy
                d = math.sqrt(cx * cx + cy * cy)
                a = rr - d + 0.5
                if a <= 0.0:
                    continue
                if a > 1.0:
                    a = 1.0
                if a > av[y, x]:
                    av[y, x] = a
                qq = d / max(rr, 0.3)
                if qq < q[y, x]:
                    q[y, x] = qq
                    inv = 1.0 / max(d, 1e-6)
                    nx[y, x] = cx * inv * min(qq, 1.0)
                    ny[y, x] = cy * inv * min(qq, 1.0)
                    sv[y, x] = (dx * py - dy * px) / L / max(rr, 0.3)
                    lv[y, x] = l0 + t * L
                    rv[y, x] = rr


@njit(cache=True, fastmath=True)
def stamp_florets(rgb, a, fl):
    """fl rows: x, y, R, phi, r, g, b, eye, ol, blob.  Straight-colour rgb + coverage a (in place).
    5-petal florets with notched petal tips, a darker pink eye and a faint outline where a floret sits over
    other blossom (so neighbouring flowers read as separate)."""
    H, W = a.shape
    for i in range(fl.shape[0]):
        cx, cy, R, ph = fl[i, 0], fl[i, 1], fl[i, 2], fl[i, 3]
        cr, cg, cb, eye, ol = fl[i, 4], fl[i, 5], fl[i, 6], fl[i, 7], fl[i, 8]
        x0 = max(0, int(cx - R - 2))
        x1 = min(W, int(cx + R + 3))
        y0 = max(0, int(cy - R - 2))
        y1 = min(H, int(cy + R + 3))
        for y in range(y0, y1):
            for x in range(x0, x1):
                dx = x + 0.5 - cx
                dy = y + 0.5 - cy
                r = math.sqrt(dx * dx + dy * dy)
                th = math.atan2(dy, dx) - ph
                c = abs(math.cos(2.5 * th))
                # petal outline: rounded petals, a small notch at each tip
                pr = R * (0.5 + 0.5 * c ** 0.55)
                pr -= R * 0.13 * max(0.0, c - 0.93) / 0.07
                blob = fl[i, 9] > 0.5
                if R < 3.0 or blob:
                    pr = R * 0.85            # tiny: a round speck / cluster body
                cov = pr - r + 0.6
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                # radial shading: slightly lighter toward the petal tips, dark pink eye
                k = 0.96 + 0.08 * min(r / max(pr, 1e-3), 1.0)
                e = 0.0
                if R >= 3.0 and not blob:
                    e = eye * max(0.0, 1.0 - r / (0.26 * R)) ** 0.8
                o = 1.0
                if blob:
                    k = 1.0
                if a[y, x] > 0.5 and R >= 3.0 and not blob:
                    o = 1.0 - ol * max(0.0, 1.0 - (pr - r) / 1.3)
                vr = cr * k * o * (1 - e) + 0.86 * e
                vg = cg * k * o * (1 - e) + 0.42 * e
                vb = cb * k * o * (1 - e) + 0.6 * e
                old = a[y, x]
                na = cov + old * (1 - cov)
                rgb[y, x, 0] = (vr * cov + rgb[y, x, 0] * old * (1 - cov)) / na
                rgb[y, x, 1] = (vg * cov + rgb[y, x, 1] * old * (1 - cov)) / na
                rgb[y, x, 2] = (vb * cov + rgb[y, x, 2] * old * (1 - cov)) / na
                a[y, x] = na


# ============================================================================ skeleton

def _bezier(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


class Tree:
    """A painted tree in its own metre-space (x right, y up, origin at trunk foot)."""

    def __init__(self, rng, lean=-1.0, scale=1.0, n_mass=(4, 6), crown=(4.0, 2.4), fork=(1.8, 2.4),
                 r_trunk=0.22, mass_r=(1.3, 1.9), droopers=(1, 3), hem=(3, 6), crown_lift=1.6, trunk_lean=0.35, clump_on_limb=0.5, sep=0.62, steep=0.0, limb_r=1.0):
        self.limbs = []           # list of (pts (n,2), radii (n,))  - drawn behind masses
        self.front = []           # branches drawn in front of the masses (drooping outer branches)
        self.masses = []          # dicts cx, cy, rx, ry  (metres)
        self.hem = []             # small clumps drawn last
        s = scale
        hf = rng.uniform(*fork) * s
        # ---------------- trunk: gentle S curve, flared foot
        top = np.array([lean * rng.uniform(0.5, 1.0) * trunk_lean * s, hf])
        c1 = np.array([-lean * rng.uniform(0.0, 0.15) * s, hf * 0.45])
        n_t = 12
        pts = _bezier(np.array([0.0, -0.2]), c1, top, n_t)
        pts[1:-1, 0] += rng.normal(0, 0.025 * s, n_t - 2)
        f = np.linspace(0, 1, n_t)
        rad = r_trunk * s * (1.0 - 0.2 * f + 0.5 * np.exp(-f / 0.07)) * (1 + 0.03 * np.sin(f * 7 + rng.uniform(0, 6)))
        self.limbs.append((pts, rad))
        self.r_trunk = r_trunk * s
        # ---------------- masses: spread over the crown (no two on top of each other), biased to the lean side
        nm = int(rng.integers(n_mass[0], n_mass[1] + 1))
        W_, H_ = crown[0] * s, crown[1] * s
        cen = []
        Rs = []
        tries = 0
        while len(cen) < nm and tries < 400:
            tries += 1
            u = rng.uniform(-1, 1)
            x = top[0] + lean * (0.25 * W_ + 0.75 * W_ * u)
            y = hf + crown_lift * s + H_ * rng.uniform(0.0, 1.0) * (1 - 0.35 * u * u)
            R = rng.uniform(*mass_r) * s
            if all(math.hypot(x - c[0], (y - c[1]) * 1.2) > sep * (R + r) for c, r in zip(cen, Rs)):
                cen.append((x, y))
                Rs.append(R)
        for (x, y), R in zip(cen, Rs):
            self.masses.append(dict(cx=float(x), cy=float(y), rx=R * rng.uniform(1.1, 1.25),
                                    ry=R * rng.uniform(0.85, 0.95), seed=int(rng.integers(1 << 30))))
        # draw order: upper masses behind, lower masses in front (crisp lit-top-against-shadow overlaps)
        order = np.argsort([-m['cy'] + rng.normal(0, 0.25 * s) for m in self.masses])
        self.masses = [self.masses[i] for i in order]
        # ---------------- primary limbs: branch off the upper trunk -> into the underside of each mass
        for i, m in enumerate(self.masses):
            j = int(rng.integers(n_t - 4, n_t))
            p0 = pts[j]
            r0 = rad[j] * rng.uniform(0.6, 0.75) * limb_r
            tgt = np.array([m['cx'] + rng.normal(0, 0.12 * m['rx']), m['cy'] + 0.1 * m['ry']])
            d = tgt - p0
            ctrl = p0 + np.array([d[0] * rng.uniform(0.35, 0.6) * (1 - steep),
                                  d[1] * (rng.uniform(0.35, 0.55) * (1 - steep) + 0.85 * steep)])
            n = 10
            P = _bezier(p0, ctrl, tgt, n)
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            P[2:-1] += nrm * rng.normal(0, 0.06 * s, (n - 3, 1))           # zig-zag of an old cherry limb
            fr = np.linspace(0, 1, n)
            self.limbs.append((P, r0 * (1 - 0.72 * fr ** 0.8)))
            ent = 8
            if rng.random() < clump_on_limb:          # a blossom clump sitting on the limb hides a segment
                kk = int(rng.integers(4, max(5, ent)))
                R = rng.uniform(0.28, 0.45) * s
                self.hem.append(dict(cx=float(P[kk, 0]), cy=float(P[kk, 1] + 0.2 * R), rx=R * 1.3, ry=R * 0.85,
                                     seed=int(rng.integers(1 << 30))))
            # secondaries forking off into the same mass
            for _ in range(int(rng.integers(1, 3))):
                k = int(rng.integers(4, 7))
                q0 = P[k]
                side = rng.choice([-1, 1])
                tg = np.array([m['cx'] + side * m['rx'] * rng.uniform(0.4, 0.8),
                               m['cy'] + m['ry'] * rng.uniform(-0.35, 0.2)])
                c = q0 + (tg - q0) * 0.5 + np.array([0, 0.25 * s])
                Q = _bezier(q0, c, tg, 7)
                self.limbs.append((Q, r0 * (1 - 0.72 * fr[k] ** 0.8) * np.linspace(0.6, 0.18, 7)))
        # ---------------- drooping outer branches (in front) with small hem clumps along them
        outer = sorted(self.masses, key=lambda m: lean * m['cx'])[::-1]
        nd = int(rng.integers(droopers[0], droopers[1] + 1))
        for j in range(nd):
            m = outer[j % len(outer)]
            q0 = np.array([m['cx'] + lean * m['rx'] * rng.uniform(0.0, 0.4), m['cy'] - m['ry'] * 0.6])
            tg = q0 + np.array([lean * rng.uniform(0.6, 1.3) * s, -rng.uniform(0.7, 1.4) * s])
            c = q0 + np.array([lean * 0.55 * s, 0.1 * s])
            Q = _bezier(q0, c, tg, 8)
            Q[1:-1] += rng.normal(0, 0.03 * s, (6, 2))
            self.limbs.append((Q, np.linspace(0.055, 0.012, 8) * s))
            for tt in (rng.uniform(0.35, 0.55), 1.0):
                k = int(round(tt * 7))
                R = rng.uniform(0.25, 0.45) * s * (1.2 if tt == 1.0 else 1.0)
                self.hem.append(dict(cx=float(Q[k, 0]), cy=float(Q[k, 1] - 0.1 * R), rx=R * 1.3, ry=R * 0.85,
                                     seed=int(rng.integers(1 << 30))))
        # hem clumps under the masses (hide limb segments, break the lower hem)
        for _ in range(int(rng.integers(hem[0], hem[1] + 1))):
            m = self.masses[int(rng.integers(len(self.masses)))]
            R = rng.uniform(0.28, 0.5) * s
            self.hem.append(dict(cx=m['cx'] + rng.uniform(-0.7, 0.7) * m['rx'],
                                 cy=m['cy'] - m['ry'] * rng.uniform(0.8, 1.0), rx=R * 1.35, ry=R * 0.85,
                                 seed=int(rng.integers(1 << 30))))

    def extent(self):
        xs, ys = [], []
        for P, r in self.limbs + self.front:
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for m in self.masses + self.hem:
            xs += [m['cx'] - m['rx'] * 1.25, m['cx'] + m['rx'] * 1.25]
            ys += [m['cy'] - m['ry'] * 1.35, m['cy'] + m['ry'] * 1.35]
        return min(xs), max(xs), min(ys), max(ys)


@njit(cache=True, fastmath=True)
def vnoise_at(u, v, seed):
    """value noise sampled at arbitrary float coordinates (2D arrays), 2 octaves, 0..1"""
    H, W = u.shape
    out = np.empty((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            acc = 0.0
            amp = 0.65
            uu, vv = u[y, x], v[y, x]
            for o in range(2):
                ix = np.int64(math.floor(uu))
                iy = np.int64(math.floor(vv))
                fx = uu - ix
                fy = vv - iy
                fx = fx * fx * (3 - 2 * fx)
                fy = fy * fy * (3 - 2 * fy)
                a = _h(ix, iy, seed)
                b = _h(ix + 1, iy, seed)
                c = _h(ix, iy + 1, seed)
                d = _h(ix + 1, iy + 1, seed)
                acc += amp * (a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy)
                amp *= 0.5
                uu = uu * 2.03 + 17.1
                vv = vv * 2.03 + 5.3
            out[y, x] = acc / 0.975
    return out


@njit(cache=True, inline='always')
def _h(ix, iy, seed):
    n = ix * np.int64(374761393) + iy * np.int64(668265263) + np.int64(seed) * np.int64(1442695041)
    n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
    n = n ^ (n >> np.int64(16))
    return (n & np.int64(0xFFFFFF)) / 16777216.0


# ============================================================================ painting

class Painter:
    """Paints a Tree onto an RGBA card (premultiplied rgb + alpha) at k px per metre."""

    def __init__(self, k, sun_dir=(-0.8, -0.6), fl_m=0.05, fl_min=1.2, lace=1.0, holes=1.0, detail=1.0,
                 bark=((0.15, 0.118, 0.118), (0.33, 0.255, 0.24)), bounce=(0.42, 0.45, 0.62), rim=(1.0, 0.76, 0.52),
                 pal=None):
        self.k = k
        L = np.array(sun_dir, np.float32)
        self.L = L / np.linalg.norm(L)
        self.fr = max(fl_min, fl_m * k)          # floret radius (px)
        self.lace = lace
        self.holes = holes
        self.detail = detail
        self.bark = [np.array(c, np.float32) for c in bark]
        self.bounce = np.array(bounce, np.float32)
        self.rimc = np.array(rim, np.float32)
        self.pal = pal or dict(lit=LIT, hot=HOT, mid=MID, lav=LAV, deep=DEEP, trans=TRANS, rim=RIM)

    # ------------------------------------------------------------------ bark
    def bark_layer(self, h, w, limbs, to_px, rng, shade_top=None):
        segs = []
        for P, r in limbs:
            pp = to_px(P)
            rr = r * self.k
            l0 = 0.0
            for i in range(len(pp) - 1):
                segs.append((pp[i, 0], pp[i, 1], rr[i], pp[i + 1, 0], pp[i + 1, 1], rr[i + 1], l0))
                l0 += float(np.hypot(*(pp[i + 1] - pp[i])))
        segs = np.array(segs, np.float64)
        q = np.full((h, w), 1e9, np.float32)
        nx = np.zeros((h, w), np.float32)
        ny = np.zeros((h, w), np.float32)
        sv = np.zeros((h, w), np.float32)
        lv = np.zeros((h, w), np.float32)
        rv = np.ones((h, w), np.float32)
        av = np.zeros((h, w), np.float32)
        limb_raster(segs, q, nx, ny, sv, lv, rv, av)
        col = np.zeros((h, w, 3), np.float32)
        iy, ix = np.nonzero(av > 0)
        if len(iy) == 0:
            return col, av
        qq = np.clip(q[iy, ix], 0, 1)
        nxp, nyp = nx[iy, ix], ny[iy, ix]
        nz = np.sqrt(np.clip(1 - nxp * nxp - nyp * nyp, 0, 1))
        lx, ly = self.L
        ndl = nxp * lx + nyp * ly                    # 2D facing toward the sun
        # vertical fissures: streaks running along the limb axis (across-coordinate x slow along-coordinate)
        n1 = rng.random(len(iy)).astype(np.float32)  # paper tooth
        rs = np.maximum(rv[iy, ix], 1.0)
        fis_u = sv[iy, ix] * np.clip(rs / 4.0, 1.5, 7.0)
        fis_v = lv[iy, ix] / np.maximum(rs * 3.0, 4.0)
        nf = vnoise_at((fis_u * 1.6).astype(np.float32)[None], (fis_v * 0.35).astype(np.float32)[None], 7)[0]
        big = rs > 2.5
        fis = np.maximum(_ss(0.6, 0.82, nf), 0.5 * _ss(0.3, 0.12, nf)) * big
        lit = np.clip(ndl, 0, 1)
        base = self.bark[0] + (self.bark[1] - self.bark[0]) * (0.3 + 0.3 * nz + 0.4 * lit)[:, None]
        c = base * (1 - 0.45 * fis)[:, None] * (0.95 + 0.1 * n1)[:, None]
        rim = _ss(0.72, 0.95, qq) * _ss(0.15, 0.65, lit / np.maximum(qq, 1e-3))
        bnc = _ss(0.45, 0.97, qq) * _ss(0.0, 0.7, -ndl / np.maximum(qq, 1e-3))
        c = c * (1 - 0.45 * bnc[:, None]) + self.bounce * (0.45 * bnc)[:, None]
        # a warm sun side + a crisp 2-3 px rim
        c = c + self.rimc * (0.1 * _ss(0.1, 0.7, lit))[:, None]
        c = c * (1 - 0.9 * rim[:, None]) + self.rimc * (0.9 * rim)[:, None]
        if shade_top is not None:
            # inside the crown the wood sits in the blossoms' lavender shade (no black specks)
            st = shade_top[iy, ix][:, None]
            c = c * (1 - st) + np.array([0.27, 0.2, 0.28], np.float32) * st
        col[iy, ix] = c
        return col, av

    # ------------------------------------------------------------------ a blossom mass
    def mass(self, card, m, to_px, rng, kind='mass', occl=None):
        """paint one blossom mass (dict in metres) into card (premult rgb + a), in place."""
        k = self.k
        H, W = card.shape[:2]
        cx, cy = to_px(np.array([[m['cx'], m['cy']]]))[0]
        rx, ry = m['rx'] * k, m['ry'] * k
        R = 0.5 * (rx + ry)
        pad = int(0.45 * R + 6 * self.fr + 4)
        X0, X1 = int(max(0, cx - rx - pad)), int(min(W, cx + rx + pad))
        Y0, Y1 = int(max(0, cy - ry - pad)), int(min(H, cy + ry + pad))
        if X1 - X0 < 3 or Y1 - Y0 < 3:
            return
        h, w = Y1 - Y0, X1 - X0
        mr = np.random.default_rng(m['seed'])
        # smooth fields (silhouette SDF, value masses, colour) are evaluated on a coarser grid and upsampled
        qf = max(1, min(4, int(R / 90)))
        hl, wl_ = (h + qf - 1) // qf, (w + qf - 1) // qf
        yy, xx = np.mgrid[0:hl, 0:wl_].astype(np.float32)
        xx = (xx + 0.5) * qf - 0.5 + (X0 - cx)
        yy = (yy + 0.5) * qf - 0.5 + (Y0 - cy)

        def up(a_):
            if qf == 1:
                return a_
            return cv2.resize(a_, (wl_ * qf, hl * qf), interpolation=cv2.INTER_LINEAR)[:h, :w]

        # ---- silhouette: union of sub-clumps (cauliflower top, flatter base) as an approximate SDF (px)
        big = kind == 'mass'
        nsub = int(mr.integers(5, 8)) if big else int(mr.integers(3, 5))
        sd = np.full((hl, wl_), 1e9, np.float32)
        subs = [(0.0, 0.1, 0.8, 0.72)]
        for i in range(nsub):
            a = math.pi * (0.02 + 0.96 * (i + mr.uniform(-0.25, 0.25)) / max(nsub - 1, 1))
            dd = mr.uniform(0.42, 0.6)
            subs.append((math.cos(a) * dd, -math.sin(a) * dd * 0.85 + 0.05, mr.uniform(0.36, 0.52),
                         mr.uniform(0.36, 0.5)))
        for j in range(2):                         # the flatter underside
            subs.append((mr.uniform(-0.4, 0.4), mr.uniform(0.32, 0.42), mr.uniform(0.45, 0.6), 0.3))
        for (ox, oy, sx, sy) in subs:
            ex, ey = sx * rx, sy * ry
            e = np.sqrt(((xx - ox * rx) / ex) ** 2 + ((yy - oy * ry) / ey) ** 2)
            sd = np.minimum(sd, (e - 1.0) * min(ex, ey))
        nz_lo = _noise(hl, wl_, max(4.0, 0.45 * R) / qf, mr) - 0.5
        nz_mid = _noise(hl, wl_, max(3.0, 0.12 * R) / qf, mr) - 0.5
        sd = sd + nz_lo * 0.18 * R + nz_mid * 0.06 * R
        fr = self.fr
        bw = max(2.4 * fr, 0.09 * R) * self.lace
        bw_l = bw * (0.35 + 1.2 * _noise(hl, wl_, max(3.0, 2.0 * bw) / qf, mr))
        # ---- colour field: big smooth value masses (dome normal from a heavily blurred silhouette)
        sig = max(1.0, 0.33 * R)
        fq = max(1.0, sig / 4.0)
        lw, lh = max(4, int(w / fq)), max(4, int(h / fq))
        sm = cv2.resize(_ss(0.0, -1.0, sd).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), sig / fq)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / (8 * fq)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / (8 * fq)
        ndl = np.clip(-(gx * self.L[0] + gy * self.L[1]) * sig * 2.5, -1, 1)
        ndl = cv2.resize(ndl, (wl_, hl), interpolation=cv2.INTER_CUBIC)
        hf = (yy + ry) / (2 * ry)                      # 0 at the top, 1 at the bottom
        v = 0.9 - 1.05 * hf + 0.45 * ndl + 0.3 * nz_lo + 0.08 * nz_mid
        if not big:
            v = v + 0.1
        pal = self.pal
        # painted value shapes: crisp-edged bands (lit / mid / lavender) + a faint gradient inside each
        wl = _ss(0.585, 0.615, v)[..., None]
        wm = _ss(0.175, 0.205, v)[..., None]
        wh = _ss(0.9, 0.93, v)[..., None]
        col = pal['lav'] * (1 - wm) + (pal['mid'] * (1 - wl) + pal['lit'] * wl) * wm
        col = col * (1 - wh) + pal['hot'] * wh
        col = col * (0.96 + 0.07 * np.clip(v, 0, 1))[..., None]
        # warm transmitted light in the thin edges (strongest in the lower hem / shadow side: backlit)
        edge = np.clip(1 + sd / (2.5 * bw), 0, 1) ** 1.4
        tr = edge * (0.35 + 0.65 * _ss(0.35, 1.0, hf)) * (1 - 0.7 * wl[..., 0])
        if not big:
            tr = np.maximum(tr, 0.3 * (1 - wl[..., 0]))
        col = col * (1 - 0.65 * tr[..., None]) + pal['trans'] * (0.65 * tr)[..., None]
        # ---- to full resolution
        sd = up(sd)
        bw_loc = up(bw_l)
        col = up(col.astype(np.float32))
        ndl = up(ndl.astype(np.float32))
        hf = up(hf.astype(np.float32))
        core = _ss(0.9, -0.9, sd + bw_loc)             # solid body, set in from the outline by the lace band
        # ---- lace: floret clusters along the band + pinholes
        fl = []
        cov = core.copy()

        def cluster(px, py, crad, body=True, nf=None):
            if body:
                fl.append([px, py, 0.62 * crad, 0.0, 1.0])
            nf_ = nf if nf is not None else int(mr.integers(3, 7))
            for _ in range(nf_):
                a_ = mr.uniform(0, 6.283)
                d_ = crad * (0.45 + 0.5 * math.sqrt(mr.uniform(0, 1)))
                fl.append([px + math.cos(a_) * d_, py + math.sin(a_) * d_, fr * mr.uniform(0.75, 1.1),
                           mr.uniform(0, 6.283), 0.0])

        # pinholes (cut the body); their rims get florets
        n_holes = 0
        if big and fr >= 1.2:
            area = float((sd < 0).sum())
            n_holes = min(7, int(round(self.holes * area / (math.pi * (1.4 * k) ** 2) * 3.0)))
        if n_holes > 0:
            cand = np.argwhere((sd < -1.0 * bw) & (sd > -2.6 * bw - 0.12 * R) & (hf > 0.2) & (hf < 0.9))
            if len(cand):
                pick = cand[mr.choice(len(cand), min(n_holes, len(cand)), replace=False)]
                for (py, px) in pick:
                    hr = max(1.5, mr.uniform(1.0, 2.4) * fr)
                    x0_, x1_ = int(max(0, px - 3 * hr)), int(min(w, px + 3 * hr + 1))
                    y0_, y1_ = int(max(0, py - 3 * hr)), int(min(h, py + 3 * hr + 1))
                    gy_, gx_ = np.mgrid[y0_:y1_, x0_:x1_]
                    hole = np.zeros(gy_.shape, np.float32)
                    for _ in range(3):
                        ox_, oy_ = mr.normal(0, 0.6 * hr, 2)
                        rr_ = hr * mr.uniform(0.5, 1.0)
                        dd = np.sqrt((gx_ + 0.5 - px - ox_) ** 2 + (gy_ + 0.5 - py - oy_) ** 2)
                        hole = np.maximum(hole, np.clip(rr_ - dd + 0.5, 0, 1))
                    cov[y0_:y1_, x0_:x1_] *= (1 - hole)
                    for j in range(int(mr.integers(3, 6))):
                        a_ = mr.uniform(0, 6.283)
                        cluster(px + math.cos(a_) * hr * 1.1, py + math.sin(a_) * hr * 1.1, 1.3 * fr, False,
                                nf=int(mr.integers(1, 3)))
        # lace clusters along the outline band (gaps between them stay open)
        if fr >= 0.7:
            cell = max(2.0, 2.8 * fr)
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            s_ = sd[iy, ix]
            bl = bw_loc[iy, ix]
            depth = np.clip((s_ + bl) / (bl + 0.2 * bw + 1e-3), 0, 1)       # 0 at the body, 1 at the outline
            keep = (s_ < 0.2 * bw) & (s_ > -bl - 0.8 * fr) & (mr.random(s_.shape) < 0.95 - 0.6 * depth)
            for px, py, dp in zip(jx[keep], jy[keep], depth[keep]):
                cluster(px, py, fr * mr.uniform(1.1, 2.0) * (1.5 - 0.6 * dp))
            # flower texture just inside the lace (sparse, low contrast): the body reads as blossoms
            if self.detail > 0 and fr >= 2.5:
                cell2 = cell * 1.5
                gy_, gx_ = np.mgrid[0:h:cell2, 0:w:cell2]
                jx = gx_ + mr.uniform(0, cell2, gx_.shape)
                jy = gy_ + mr.uniform(0, cell2, gy_.shape)
                ix = np.clip(jx.astype(int), 0, w - 1)
                iy = np.clip(jy.astype(int), 0, h - 1)
                s_ = sd[iy, ix]
                bl = bw_loc[iy, ix]
                dep = np.maximum(-s_ - bl, 0)
                keep = (s_ < -bl) & (mr.random(s_.shape) < (0.07 + 0.33 * np.exp(-dep / (2.5 * bw))) * self.detail)
                for px, py in zip(jx[keep], jy[keep]):
                    for _ in range(int(mr.integers(1, 4))):
                        fl.append([px + mr.normal(0, fr), py + mr.normal(0, fr), fr * mr.uniform(0.7, 1.0),
                                   mr.uniform(0, 6.283), 0.0])
        # ---- assemble straight colour + coverage, then florets
        rgb = np.ascontiguousarray(col.astype(np.float32))
        a = np.ascontiguousarray(cov.astype(np.float32))
        if fl:
            F = np.array(fl, np.float64)
            ix = np.clip(F[:, 0].astype(int), 0, w - 1)
            iy = np.clip(F[:, 1].astype(int), 0, h - 1)
            c = col[iy, ix].astype(np.float64)
            isb = F[:, 4:5] > 0.5
            vv = np.where(isb, 1.0, mr.uniform(0.97, 1.05, (len(F), 1)))
            c = np.clip(c * vv, 0, 1.1)
            eye = np.where(isb, 0.0, 0.28)
            ol = np.where(isb, 0.0, 0.12)
            F = np.hstack([F[:, :4], c, eye, ol, F[:, 4:5]])
            # bodies first, then florets back to front (lower = shadowed first)
            order = np.lexsort((-F[:, 1], -F[:, 9]))
            stamp_florets(rgb, a, np.ascontiguousarray(F[order]))
        # ---- 2 px sun-side rim on the silhouette
        rimw = min(2.6, max(1.2, fr * 0.4))
        sh = _shift(a, self.L[0] * rimw * 1.5, self.L[1] * rimw * 1.5)
        rim = np.clip(a - sh, 0, 1) * _ss(-0.2, 0.4, ndl + 0.4 * (1 - hf))
        rgb = rgb * (1 - 0.7 * rim[..., None]) + pal['rim'] * (0.7 * rim)[..., None]
        # ---- cast shadow of this mass onto what is already painted behind it (crisp, lavender)
        sub = card[Y0:Y1, X0:X1]
        if big:
            off = 0.12 * R
            shd = _shift(a, -self.L[0] * off, -self.L[1] * off * 1.2)
            shd = cv2.GaussianBlur(shd, (0, 0), max(0.8, 0.01 * R)) * (1 - a)
            ex = occl[Y0:Y1, X0:X1] if occl is not None else sub[..., 3]
            dk = (shd * ex * 0.55)[..., None]
            sub[..., :3] = sub[..., :3] * (1 - dk) + pal['deep'] * sub[..., 3:4] * dk * 0.9
        # ---- over
        sub[..., :3] = rgb * a[..., None] + sub[..., :3] * (1 - a[..., None])
        sub[..., 3] = a + sub[..., 3] * (1 - a)
        if occl is not None:
            occl[Y0:Y1, X0:X1] = np.maximum(occl[Y0:Y1, X0:X1], a)

    # ------------------------------------------------------------------ full tree
    def paint(self, tree, rng, ss_margin=4):
        k = self.k
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * ss_margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * ss_margin
        ox = -x0 * k + ss_margin                   # card px of the trunk foot
        oy = y1 * k + ss_margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        card = np.zeros((H, W, 4), np.float32)
        # bark behind the masses: darkened up where it disappears into the canopy shade
        ys = np.arange(H, dtype=np.float32)[:, None]
        cb = min(m['cy'] - m['ry'] for m in tree.masses)
        yb = oy - cb * k
        shade = 0.85 * _ss(yb + 0.3 * k, yb - 0.6 * k, ys) * np.ones((1, W), np.float32)
        col, av = self.bark_layer(H, W, tree.limbs, to_px, rng, shade.astype(np.float32))
        card[..., :3] = col * av[..., None]
        card[..., 3] = av
        occl = np.zeros((H, W), np.float32)
        for m in tree.masses:
            self.mass(card, m, to_px, rng, 'mass', occl)
        if tree.front:
            col, av = self.bark_layer(H, W, tree.front, to_px, rng)
            card[..., :3] = col * av[..., None] + card[..., :3] * (1 - av[..., None])
            card[..., 3] = av + card[..., 3] * (1 - av)
        # blossom clumps where each limb disappears into a mass: the wood never ends in a hard cut
        entry = []
        for P, r in tree.limbs[1:]:
            pp = to_px(P)
            n = len(pp)
            tt = np.linspace(0, 1, 4 * n)
            xs = np.interp(tt, np.linspace(0, 1, n), pp[:, 0])
            ys = np.interp(tt, np.linspace(0, 1, n), pp[:, 1])
            rs = np.interp(tt, np.linspace(0, 1, n), r)
            ix = np.clip(xs.astype(int), 0, W - 1)
            iy = np.clip(ys.astype(int), 0, H - 1)
            inside = occl[iy, ix] > 0.6
            if not inside.any() or inside[0]:
                continue
            j = int(np.argmax(inside))
            R = max(3.2 * rs[j], 0.28)
            cxm, cym = (xs[j] - ox) / k, (oy - ys[j]) / k
            entry.append(dict(cx=float(cxm), cy=float(cym - 0.05 * R), rx=R * 1.3, ry=R * 0.9,
                              seed=int(rng.integers(1 << 30))))
        for m in entry + tree.hem:
            self.mass(card, m, to_px, rng, 'hem', None)
        # the trunk foot enters the ground: nothing below the ground line
        cut = _ss(oy + 0.03 * k + 1.0, oy + 0.03 * k - 1.0, np.arange(H, dtype=np.float32))[:, None]
        card *= cut[..., None]
        return card, ox, oy


def haze_card(card, hz, col):
    """blend a premultiplied card toward the aerial haze colour"""
    if hz <= 0:
        return card
    c = np.asarray(col, np.float32)
    card[..., :3] = card[..., :3] * (1 - hz) + c * card[..., 3:4] * hz
    return card


def card_to_screen(card, ox, oy, bx, by, ss):
    """premultiplied ss-card whose trunk foot is at card px (ox, oy) -> straight RGBA at screen res placed so
    the foot lands on screen (bx, by).  Returns rgba, x0, y0 (integer screen origin)."""
    X0 = bx * ss - ox
    Y0 = by * ss - oy
    ix, iy = int(math.floor(X0 / ss)), int(math.floor(Y0 / ss))
    fx, fy = X0 - ix * ss, Y0 - iy * ss                # sub-pixel remainder in ss px
    h, w = card.shape[:2]
    Wn, Hn = int(math.ceil((w + fx) / ss)) + 1, int(math.ceil((h + fy) / ss)) + 1
    M = np.float32([[1, 0, fx], [0, 1, fy]])
    big = cv2.warpAffine(card, M, (Wn * ss, Hn * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    sm = cv2.resize(big, (Wn, Hn), interpolation=cv2.INTER_AREA)
    a = sm[..., 3:4]
    rgb = sm[..., :3] / np.maximum(a, 1e-4)
    return np.ascontiguousarray(np.concatenate([rgb, a], -1).astype(np.float32)), ix, iy
