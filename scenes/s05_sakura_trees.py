"""s05_sakura helpers: procedural Somei-Yoshino cherry trees as 3D primitives.

A tree = trunk + curved scaffold limbs + two levels of sub-branches (bark capsules) carrying blossom
CLUMPS: a small core dome, a ring of CHILD domes on its outer/upper/viewer-facing surface, and dense tiny
FLORET domes covering each child (individual flower bunches -> scalloped, individually lit edges). Loose
twigs with little floret bunches poke out of the silhouettes. Primitives live in the world path frame
(u, Y, s) and are projected by `project()` into screen-space rows for s05_sakura_raster.zbuffer.
"""
import math
import numpy as np

from s05_sakura_raster import (NP, K_KIND, K_X0, K_Y0, K_Z0, K_R0, K_X1, K_Y1, K_Z1, K_R1, K_AY, K_RZ0, K_RZ1,
                               K_PAR, K_TREE, K_MAT, K_TONE)


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / (math.sqrt(float((v * v).sum())) + 1e-9)


def _unit_rows(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class TreeBuilder:
    def __init__(self):
        self.sph = []     # arrays (n, 9): u, Y, s, r, ay, parent, tree, mat, tone
        self.cap = []     # (u0, Y0, s0, r0, u1, Y1, s1, r1, tree)
        self.n = 0

    def spheres(self, c, r, ay, parent, tree, mat, tone):
        c = np.atleast_2d(c)
        n = len(c)
        a = np.zeros((n, 9))
        a[:, 0:3] = c
        a[:, 3] = r
        a[:, 4] = ay
        a[:, 5] = parent
        a[:, 6] = tree
        a[:, 7] = mat
        a[:, 8] = tone
        self.sph.append(a)
        idx = np.arange(self.n, self.n + n)
        self.n += n
        return idx

    def capsule(self, p0, p1, r0, r1, tree, mat=2):
        # arc length along chained capsules (continuous bark texture across joints)
        if not hasattr(self, '_ends'):
            self._ends = {}
        k0 = (round(float(p0[0]), 5), round(float(p0[1]), 5), round(float(p0[2]), 5))
        a0 = self._ends.get(k0, (abs(float(p0[0])) * 7.3 + abs(float(p0[2])) * 3.1) % 50.0)
        L = float(np.linalg.norm(np.asarray(p1, np.float64) - np.asarray(p0, np.float64)))
        k1 = (round(float(p1[0]), 5), round(float(p1[1]), 5), round(float(p1[2]), 5))
        self._ends[k1] = a0 + L
        self.cap.append((p0[0], p0[1], p0[2], r0, p1[0], p1[1], p1[2], r1, tree, a0, L, mat))


def _mix(rng, mix):
    """scale multiplier drawn from a few size classes: mix = ((prob, lo, hi), ...)"""
    u = rng.random()
    acc = 0.0
    for (p_, lo, hi) in mix:
        acc += p_
        if u < acc:
            return rng.uniform(lo, hi)
    return rng.uniform(mix[-1][1], mix[-1][2])


def _hemi(rng, n, bias, spread=1.0):
    v = rng.normal(0, spread, (n, 3)) + bias
    return _unit_rows(v)


def clump(B, rng, tid, c, rc, toward, out, floret=True, detail=1.0, tone=0.0, up=0.7, twigs=True, fl_scale=1.0):
    """One blossom clump at c (radius rc).  Florets/children favour the side facing the camera (origin)."""
    toward = _norm(-np.asarray(c, np.float64))
    cid = B.spheres(c, rc * 0.72, rng.uniform(0.75, 0.88), -1, tid, 0, tone)[0]
    nch = int(rng.integers(8, 13) * (0.7 + 0.3 * detail))
    bias = out * 0.9 + np.array([0, up, 0]) + toward * 0.6
    v = _hemi(rng, nch, bias)
    rch = rc * rng.uniform(0.14, 0.62, nch)
    cc = c + v * (rc * rng.uniform(0.55, 0.85, nch))[:, None]
    tch = tone + rng.normal(0, 0.35, nch)
    chid = B.spheres(cc, rch, rng.uniform(0.78, 0.92, nch), cid, tid, 0, tch)
    if floret:
        rf_base = 0.22 * fl_scale
        nf = int(np.clip((1.0 / rf_base) ** 2 * 1.9 * detail, 8, 48))
        par = np.repeat(np.arange(nch), nf)
        vv = _unit_rows(rng.normal(0, 1.0, (len(par), 3)) + v[par] * 0.9 + toward * 0.6 + np.array([0, 0.2, 0]))
        # strongly varied floret scale: mostly small bunches, some mid-size lobes
        _u = rng.random(len(par))
        frng = np.random.default_rng(int(abs(c[0]) * 1000 + abs(c[2]) * 777 + abs(c[1]) * 31) % (2 ** 31))
        big = frng.random(len(par)) < 0.22
        rf = rch[par] * np.where(big, frng.uniform(0.24, 0.4, len(par)), frng.uniform(0.07, 0.17, len(par))) * fl_scale
        fc = cc[par] + vv * (rch[par] * rng.uniform(0.8, 1.0, len(par)))[:, None]
        B.spheres(fc, rf, rng.uniform(0.8, 1.0, len(par)), chid[par], tid, 1, tch[par] + rng.normal(0, 0.55, len(par)))
    if twigs and floret:
        # loose twigs poking out of the silhouette with a little bunch of flowers at the end
        for k in range(int(rng.integers(1, 4))):
            d = _norm(rng.normal(0, 1, 3) + out * 1.2 + np.array([0, -0.3, 0]) + toward * 0.4)
            p0 = c + d * rc * 0.75
            p1 = c + d * rc * rng.uniform(1.08, 1.3)
            B.capsule(p0, p1, 0.008, 0.004, tid)
            nb = int(rng.integers(4, 9))
            fb = p1 + rng.normal(0, rc * 0.09, (nb, 3))
            B.spheres(fb, rc * rng.uniform(0.07, 0.11, nb), 0.9, -1, tid, 1, tone + rng.normal(0, 0.5, nb))
    return cid


def tuft(B, rng, tid, c, rt, tone=0.0):
    """Small bunch of flowers sitting directly on a branch (hides branch ends, dots the limbs)."""
    c = np.asarray(c, np.float64)
    toward = _norm(-c)
    core = B.spheres(c, rt * 0.7, rng.uniform(0.7, 0.9), -1, tid, 0, tone)[0]
    n = int(rng.integers(7, 13))
    v = _unit_rows(rng.normal(0, 1, (n, 3)) + toward * 0.7 + np.array([0, 0.25, 0]))
    pos = c + v * (rt * rng.uniform(0.55, 0.85, n))[:, None]
    B.spheres(pos, rt * rng.uniform(0.28, 0.42, n), rng.uniform(0.8, 1.0, n), core, tid, 1,
              tone + rng.normal(0, 0.5, n))


def grow_tree(B, rng, tid, base, height=1.0, lean=(0.0, 0.0), spread=1.0, detail=1.0, floret=True,
              toward=(0.0, 0.0, -1.0), clump_scale=1.0, droop=1.0, min_clump_h=2.3, lean_amt=0.6, fl_scale=1.0,
              twigs=True, trunk=(2.5, 3.1), el_rng=(0.5, 1.0), limb_r=0.66, tip_r=(0.35, 0.25), child_r=0.7,
              limb_len=(3.4, 4.6), clump_rng=(0.5, 0.75), tip_keep=0.66, tufts=0.0,
              tuft_step=0.45, tuft_r=(0.14, 0.24), tuft_min_h=1.4, n_limbs=(4, 7),
              tuft_start=(0.55, 0.3, 0.15), grass=0, collar=True, roots=False, flare=1.0, zigzag=0.0,
              twig_len=1.0, clump_mix=None, end_tuft=False):
    """base: (u, Y, s) trunk foot.  lean: (du, ds) horizontal direction the crown extends toward.
    toward: world direction toward the viewer (florets / children favour the visible side)."""
    base = np.asarray(base, np.float64)
    lean = np.array([lean[0], 0.0, lean[1]])
    toward = _norm(toward)
    hs = height
    tr_h = rng.uniform(*trunk) * hs
    top = base + np.array([0, tr_h, 0]) + lean * lean_amt * hs
    r_base = 0.19 * hs
    # root flare: short buttress roots spreading into the ground
    nroot = int(rng.integers(4, 6))
    a0 = rng.uniform(0, 6.28)
    rr_ = np.random.default_rng(tid * 131 + 17)      # own stream: keeps the crown layout stable
    for k in range(nroot):
        a = a0 + k * 6.283 / nroot + rng.normal(0, 0.3)
        d = np.array([math.cos(a), 0.0, math.sin(a)])
        # low buttress root: leaves the flared trunk foot and dives into the ground
        p0 = base + np.array([0, rr_.uniform(0.2, 0.32) * hs, 0]) + d * r_base * 0.4
        _ = rng.uniform(1.4, 1.8)
        p1 = base + d * r_base * rr_.uniform(1.25, 1.6) + np.array([0, -0.12, 0])
        if roots:
            B.capsule(p0, p1, r_base * 0.5, r_base * 0.16, tid)
    # trunk: a gentle S-curve (cubic bezier, the two controls bowed to opposite sides) with irregular
    # girth: burls, a slight waist, and a soft flare into the ground (no separate root 'claws')
    bow = rng.normal(0, 0.18 * hs, 3) * np.array([1, 0, 1]) - lean * 0.12 * hs
    side = _norm(np.cross(np.array([0, 1.0, 0]), np.asarray(toward) + 1e-6))
    sgn = 1.0 if rr_.random() < 0.5 else -1.0
    ctrl1 = base + (top - base) * 0.3 + bow * 0.6 + side * sgn * rr_.uniform(0.14, 0.24) * hs
    ctrl2 = base + (top - base) * 0.72 + bow - side * sgn * rr_.uniform(0.16, 0.28) * hs
    ctrl = ctrl2
    npt = 26
    P0 = base - np.array([0, 0.3, 0])

    def trunk_at(f_):
        return (1 - f_) ** 3 * P0 + 3 * (1 - f_) ** 2 * f_ * ctrl1 + 3 * (1 - f_) * f_ ** 2 * ctrl2 + f_ ** 3 * top

    prev = P0
    ph = rng.uniform(0, 6.28)
    bumps = [(rr_.uniform(0.15, 0.95), rr_.uniform(-0.09, 0.13), rr_.uniform(0.04, 0.1)) for _ in range(5)]
    prev_r = r_base * (1.12 + 0.3 * flare)
    for k in range(1, npt + 1):
        f_ = k / npt
        q = (1 - f_) ** 3 * P0 + 3 * (1 - f_) ** 2 * f_ * ctrl1 + 3 * (1 - f_) * f_ ** 2 * ctrl2 + f_ ** 3 * top
        g_ = 1.0 + 0.05 * math.sin(f_ * 9.0 + ph) + 0.03 * math.sin(f_ * 23.0 + 2 * ph)
        for (bc, ba, bw) in bumps:
            g_ += ba * math.exp(-((f_ - bc) / bw) ** 2)
        rq = r_base * (1.1 - 0.32 * f_ - 0.2 * f_ ** 4 + flare * 0.42 * math.exp(-f_ / 0.06)) * g_
        B.capsule(prev, q, prev_r, rq, tid)
        prev, prev_r = q, rq
    # grass tufts hugging the trunk foot (roots flare into the grass)
    if grass > 0:
        ng = int(grass)
        rg = np.random.default_rng(tid * 7 + 1001)
        for k in range(ng):
            a = rg.uniform(0, 6.283)
            rad = r_base * rg.uniform(0.7, 2.6)
            g0 = base + np.array([math.cos(a) * rad, -0.03, math.sin(a) * rad])
            hgt = rg.uniform(0.06, 0.2) * (1.3 if rad < r_base * 1.4 else 1.0)
            lean_g = rg.normal(0, 0.05, 3) * np.array([1, 0, 1])
            g1 = g0 + np.array([0, hgt, 0]) + lean_g
            B.capsule(g0, g1, rg.uniform(0.006, 0.01), 0.0015, tid, mat=3)
    tips = []

    def limb(p, d, length, r0, level):
        n = 6
        pts = [p]
        dcur = _norm(d)
        seg = length / n
        dro = (0.03, 0.09, 0.2)[level] * droop
        zr = np.random.default_rng(int(abs(p[0]) * 7919 + abs(p[2]) * 104729 + abs(p[1]) * 31 + level) % (2 ** 31))
        zs = 1.0 if zr.random() < 0.5 else -1.0
        for k in range(n):
            dcur = _norm(dcur + np.array([0, -dro * (k + 1) / n, 0]) + rng.normal(0, 0.1, 3))
            if zigzag > 0:
                # cherry wood grows in short zig-zag segments between the buds: alternate a kink sideways
                sd_ = np.cross(dcur, np.array([0, 1.0, 0]))
                if np.linalg.norm(sd_) < 1e-3:
                    sd_ = np.array([1.0, 0, 0])
                sd_ = _norm(sd_ + 0.4 * zr.normal(0, 1, 3))
                zs = -zs
                dk = _norm(dcur + sd_ * zs * zigzag * zr.uniform(0.6, 1.3))
                pts.append(pts[-1] + dk * seg)
            else:
                pts.append(pts[-1] + dcur * seg)
        tipf = tip_r[1] if level == 2 else tip_r[0]
        ff_ = np.linspace(0, 1, n + 1)
        rr = r0 * (tipf + (1 - tipf) * (1 - ff_) ** (2.3 if level == 0 else 1.5))      # tapers quickly: no thick sawn-off ends
        if level == 0 and collar:
            # collar: blend the scaffold limb smoothly out of the trunk (no broken-geometry kink)
            c0 = p - np.array([0, 0.35 * hs, 0])
            B.capsule(c0, pts[1], r_base * 0.72, rr[1], tid)
        for k in range(n):
            B.capsule(pts[k], pts[k + 1], rr[k], rr[k + 1], tid)
        # taper the limb end into thin twigs that fade into the blossom (no blunt cut ends)
        trng = np.random.default_rng(int(abs(pts[-1][0]) * 1000 + abs(pts[-1][2]) * 7919) % (2 ** 31))
        tp = pts[-1]
        td = dcur
        tr0 = rr[-1]
        nt = 3
        tl = seg * (0.9 if level == 0 else 0.7) * twig_len
        for k in range(nt):
            td = _norm(td + trng.normal(0, 0.25, 3) + np.array([0, 0.05, 0]))
            q = tp + td * tl * (1.0 - 0.2 * k)
            r1_ = max(tr0 * (1.0 - (k + 1) / nt) ** 1.3, 0.004)
            B.capsule(tp, q, max(tr0, 0.004), r1_, tid)
            if k == 0 and trng.random() < 0.7:
                # a forked side twig
                sd = _norm(td + trng.normal(0, 0.8, 3))
                B.capsule(tp, tp + sd * tl * 0.7, max(tr0 * 0.6, 0.004), 0.003, tid)
            tp, tr0 = q, r1_
        if end_tuft and floret:
            # every limb end disappears into a bunch of blossom (no bare stick tips against the sky)
            rt = rng.uniform(*tuft_r) * hs * (1.5 if level == 0 else 1.15)
            tuft(B, rng, tid, tp + np.array([0, rt * 0.2, 0]), rt, tone=rng.uniform(-1, 1))
        if tufts > 0 and floret:
            # flowers bunched directly along the outer part of the smaller branches
            ptsa = np.array(pts)
            segl = np.linalg.norm(np.diff(ptsa, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(segl)])
            start = tuft_start[level] * cum[-1]
            step = tuft_step * (1.4 if level == 0 else 1.0)
            dd = start + rng.uniform(0, step)
            while dd < cum[-1] + 0.05:
                if rng.random() < tufts:
                    q = np.array([np.interp(dd, cum, ptsa[:, j]) for j in range(3)])
                    if q[1] - base[1] > tuft_min_h * hs:
                        rt = rng.uniform(*tuft_r) * hs * (1.25 if level == 0 else 1.0)
                        tuft(B, rng, tid, q + rng.normal(0, rt * 0.3, 3) + np.array([0, rt * 0.2, 0]), rt,
                             tone=rng.uniform(-1, 1))
                dd += step * rng.uniform(0.7, 1.3)
        if level < 2:
            nchild = int(rng.integers(3, 5)) if level == 0 else int(rng.integers(2, 4))
            for c in range(nchild):
                fpos = rng.uniform(0.3, 0.92)
                k = min(int(fpos * n), n - 1)
                q = pts[k] + (pts[k + 1] - pts[k]) * (fpos * n - k)
                side = _norm(np.cross(dcur, np.array([0, 1.0, 0])) + 1e-6)
                ang = rng.uniform(0.45, 1.0) * (1 if c % 2 else -1)
                nd = _norm(dcur * math.cos(ang) + side * math.sin(ang) + np.array([0, rng.uniform(0.0, 0.4), 0]))
                limb(q, nd, length * rng.uniform(0.5, 0.7), rr[k] * child_r, level + 1)
        k0 = 2 if level == 2 else 3
        for k in range(k0, n + 1):
            tips.append((pts[k], level, dcur, k == n))

    nl = int(rng.integers(*n_limbs))
    az0 = math.atan2(lean[2], lean[0]) if np.linalg.norm(lean) > 0 else rng.uniform(0, 6.28)
    for i in range(nl):
        az = az0 + (i - (nl - 1) / 2) * (4.2 / nl) * spread + rng.normal(0, 0.25)
        el = rng.uniform(*el_rng)
        d = np.array([math.cos(az) * math.cos(el), math.sin(el), math.sin(az) * math.cos(el)])
        L = rng.uniform(*limb_len) * hs * spread ** 0.5
        org = trunk_at(rng.uniform(0.78, 1.0)) + rng.normal(0, 0.03, 3) * np.array([1, 0, 1])
        limb(org, d, L, r_base * limb_r, 0)

    crown_c = base + np.array([0, tr_h * 1.7, 0]) + lean * hs * 1.2
    for (p, level, d, final) in tips:
        if not final and rng.random() > tip_keep * detail + 0.2:
            continue
        hloc = p[1] - base[1]
        if not final and hloc < min_clump_h * hs and rng.random() > 0.12:
            continue
        if final and hloc < min_clump_h * hs * 0.75:
            continue
        rc = rng.uniform(*clump_rng) * hs * clump_scale
        if clump_mix is not None:
            rc *= _mix(rng, clump_mix)
        c = p + np.array([0, rc * 0.3, 0]) + rng.normal(0, 0.12, 3)
        out = _norm(c - crown_c)
        clump(B, rng, tid, c, rc, toward, out, floret=floret, detail=detail, tone=rng.uniform(-1, 1),
              fl_scale=fl_scale, twigs=twigs)
    # leader: the trunk continues into a tapering central limb (no capped trunk top between the limbs)
    rng = np.random.default_rng(tid * 977 + 5)
    tips.clear()
    dlead = _norm(_norm(top - ctrl) + np.array([0, 0.6, 0]) + lean * 0.4)
    limb(trunk_at(0.95), dlead, rng.uniform(*limb_len) * hs * 0.75, rq * 0.95, 0)
    for (p, level, d, final) in tips:
        if not final and rng.random() > tip_keep * detail + 0.2:
            continue
        hloc = p[1] - base[1]
        if hloc < min_clump_h * hs * 0.75:
            continue
        rc = rng.uniform(*clump_rng) * hs * clump_scale
        if clump_mix is not None:
            rc *= _mix(rng, clump_mix)
        c = p + np.array([0, rc * 0.3, 0]) + rng.normal(0, 0.12, 3)
        out = _norm(c - crown_c)
        clump(B, rng, tid, c, rc, toward, out, floret=floret, detail=detail, tone=rng.uniform(-1, 1),
              fl_scale=fl_scale, twigs=twigs)
    return B


def flower_bunch(B, rng, tid, c, rc, nfl, rf=(0.016, 0.022), out=None):
    """Close-up bunch of individual flowers around a small dark-pink core."""
    c = np.asarray(c, np.float64)
    toward = _norm(-c)
    core = B.spheres(c, rc * 0.5, 0.85, -1, tid, 0, rng.uniform(-1, -0.3))[0]
    bias = toward * 0.8 + (out if out is not None else 0.0)
    v = _unit_rows(rng.normal(0, 1, (nfl, 3)) + bias)
    pos = c + v * (rc * rng.uniform(0.75, 1.0, nfl))[:, None]
    B.spheres(pos, rng.uniform(rf[0], rf[1], nfl), rng.uniform(0.75, 1.0, nfl), core, tid, 1,
              rng.normal(0.3, 0.6, nfl))


def hero_branch(B, rng, tid, pts_world, r0=0.05, step=0.06, rc=(0.04, 0.065), toward=(0, 0, -1)):
    """Hand-placed hanging branch (world points) covered with flower bunches on short spurs."""
    pts = np.asarray(pts_world, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    sc = np.concatenate([[0], np.cumsum(seg)])
    Ltot = sc[-1]
    ts = np.linspace(0, Ltot, 60)
    P = np.stack([np.interp(ts, sc, pts[:, k]) for k in range(3)], 1)
    for k in range(3):
        P[:, k] = np.convolve(np.pad(P[:, k], 2, mode='edge'), np.ones(5) / 5, mode='valid')
    # gnarl: a few kinks (cherry wood zig-zags between buds)
    kink = np.zeros_like(P)
    for j in range(7):
        c0 = rng.uniform(0.1, 0.95) * len(P)
        amp = rng.normal(0, 0.035, 3)
        kink += amp * np.exp(-((np.arange(len(P)) - c0) / 3.0) ** 2)[:, None]
    P += kink
    rr = np.linspace(r0, r0 * 0.25, len(P))
    for k in range(len(P) - 1):
        B.capsule(P[k], P[k + 1], rr[k], rr[k + 1], tid)
    d_all = np.diff(P, axis=0)
    acc = 0.0
    for k in range(len(P) - 1):
        acc += np.linalg.norm(d_all[k])
        if ts[k] < Ltot * 0.28 or acc < step:
            continue
        acc = 0.0
        tan = _norm(d_all[k])
        side = _norm(np.cross(tan, rng.normal(0, 1, 3)))
        # short spur with one or two bunches
        L = rng.uniform(0.03, 0.12)
        p1 = P[k] + side * L + np.array([0, -0.02, 0])
        B.capsule(P[k], p1, rr[k] * 0.35, 0.004, tid)
        rcl = rng.uniform(*rc)
        flower_bunch(B, rng, tid, p1 + side * rcl * 0.5, rcl, int(rng.integers(8, 15)), out=side * 0.6)
        if rng.random() < 0.5:
            q = P[k] - side * rng.uniform(0.02, 0.06)
            flower_bunch(B, rng, tid, q, rcl * 0.8, int(rng.integers(5, 10)), out=-side * 0.6)
    flower_bunch(B, rng, tid, P[-1] + _norm(d_all[-1]) * 0.03, rc[1], 14, out=_norm(d_all[-1]))


def project(B, cam, mx, ss, zmin=0.3):
    """-> primitive table P (n, NP) in ss canvas px (camera at tx = 0), plus camera-space Z per row."""
    sph = np.concatenate(B.sph, 0) if B.sph else np.zeros((0, 9))
    cap = np.array(B.cap, np.float64).reshape(-1, 12)
    f = cam.f
    rows = []
    zs = []
    if len(sph):
        X, Y, Z = cam.to_cam(sph[:, 0], sph[:, 1], sph[:, 2])
        ok = Z > zmin
        x, y = cam.proj(X, Y, Z, 0.0, mx, ss)
        r = f * sph[:, 3] / np.maximum(Z, 1e-3) * ss
        P = np.zeros((len(sph), NP))
        P[:, K_KIND] = 0
        P[:, K_X0] = x
        P[:, K_Y0] = y
        P[:, K_Z0] = Z
        P[:, K_R0] = r
        P[:, K_AY] = sph[:, 4]
        P[:, K_RZ0] = sph[:, 3]
        P[:, K_PAR] = sph[:, 5]
        P[:, K_TREE] = sph[:, 6]
        P[:, K_MAT] = sph[:, 7]
        P[:, K_TONE] = sph[:, 8]
        P[~ok, K_R0] = 0
        rows.append(P)
        zs.append(Z)
    if len(cap):
        X0, Y0, Z0 = cam.to_cam(cap[:, 0], cap[:, 1], cap[:, 2])
        X1, Y1, Z1 = cam.to_cam(cap[:, 4], cap[:, 5], cap[:, 6])
        ok = (Z0 > zmin) & (Z1 > zmin)
        x0, y0 = cam.proj(X0, Y0, Z0, 0.0, mx, ss)
        x1, y1 = cam.proj(X1, Y1, Z1, 0.0, mx, ss)
        P = np.zeros((len(cap), NP))
        P[:, K_KIND] = 1
        P[:, K_X0] = x0
        P[:, K_Y0] = y0
        P[:, K_Z0] = Z0
        P[:, K_R0] = f * cap[:, 3] / np.maximum(Z0, 1e-3) * ss
        P[:, K_X1] = x1
        P[:, K_Y1] = y1
        P[:, K_Z1] = Z1
        P[:, K_R1] = f * cap[:, 7] / np.maximum(Z1, 1e-3) * ss
        P[:, K_RZ0] = cap[:, 3]
        P[:, K_RZ1] = cap[:, 7]
        P[:, K_TONE] = cap[:, 9]
        P[:, K_AY] = cap[:, 10]
        P[:, K_PAR] = -1
        P[:, K_TREE] = cap[:, 8]
        P[:, K_MAT] = cap[:, 11]
        P[~ok, K_R0] = 0
        P[~ok, K_R1] = 0
        rows.append(P)
        zs.append(np.maximum(Z0, Z1))     # bark binned by its far end (never drawn over its own tip clump)
    P = np.concatenate(rows, 0)
    Z = np.concatenate(zs, 0)
    return P, Z, len(sph)
