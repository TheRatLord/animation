"""Hero cumulonimbus for s02_railway_crossing (round 5): painted thunderhead.

Geometry: a massif of cauliflower towers (main tower + lower shoulders). Lobes are generated on the FRONT
SURFACE of each tower's envelope (elliptic cross-section), their size falling off with height, and pushed
outward at the silhouette so the outline is a bumpy cauliflower tower. Child lobes (and tiny grandchild
bumps) sit on the upward / sun-facing surfaces of their parents.

Rendering (numba): a SMOOTH UNION of the lobe spheroids. Per pixel, every covering lobe contributes its
surface normal with weight exp((z_i - z_max) / (k r_i)), so where lobes nest the normals blend into soft
creases (no hard per-disc crescents), while a lobe standing well in front keeps its crisp outline.

Painting: e = N.L (directional, sun upper-right) blended with the whole-massif form normal; colour ramp
deep lavender core -> lavender -> lilac half-tone -> warm white (#FFF8EC) -> hot white; soft occlusion that
builds up where lobes nest (height-field cavity); sky fill on upward shadow faces; warm ground bounce on
the downward faces of the lowest lobes; thin-edge translucency; a crisp 2-3 px silver rim on the sun-side
silhouette; torn wisps under the flat base.

Anvil: a density band (lit billowy top edge, soft gradient underside) with fibrous streaks that trail
downwind into cirrus and dissolve into the sky - no hard straight edges.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


def hx(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def ss_(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gblur(a, s):
    if s < 0.3:
        return a
    return cv2.GaussianBlur(a, (0, 0), s)


def fblur(a, s):
    """Large gaussian blur via downsample."""
    if s < 6:
        return gblur(a, s)
    q = int(max(2, s // 3))
    h, w = a.shape[:2]
    sm = cv2.resize(a, (max(w // q, 2), max(h // q, 2)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), s / q)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


# lobe table: cx, cy, cz, r, ay, clip, p1, p2, wob, level, tone
NCOL = 11


@njit(cache=True, parallel=True, fastmath=True)
def _smooth_union(Hs, Ws, S, dk, kk):
    n = S.shape[0]
    cov = np.zeros((Hs, Ws), np.float32)
    hz = np.zeros((Hs, Ws), np.float32)
    nx = np.zeros((Hs, Ws), np.float32)
    ny = np.zeros((Hs, Ws), np.float32)
    nz = np.zeros((Hs, Ws), np.float32)
    lv = np.zeros((Hs, Ws), np.float32)
    tn = np.zeros((Hs, Ws), np.float32)
    for y in prange(Hs):
        yf = y + 0.5
        zmax = np.full(Ws, -1e12, np.float32)
        # pass 1: z max
        for i in range(n):
            cx, cy, cz, r, ay, clip = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4], S[i, 5]
            ry = r * ay * 1.12
            if yf < cy - ry or yf > cy + ry or yf > clip:
                continue
            rx = r * 1.12
            x0 = max(0, int(cx - rx))
            x1 = min(Ws, int(cx + rx) + 1)
            p1, p2, wob = S[i, 6], S[i, 7], S[i, 8]
            for x in range(x0, x1):
                xf = x + 0.5
                u = (xf - cx) / r
                v = (yf - cy) / (r * ay)
                th = math.atan2(v, u)
                rr = 1.0 + wob * (0.6 * math.sin(3.0 * th + p1) + 0.4 * math.sin(5.0 * th + p2))
                d2 = (u * u + v * v) / (rr * rr)
                if d2 >= 1.0:
                    continue
                z = cz + r * math.sqrt(1.0 - d2) * dk
                if z > zmax[x]:
                    zmax[x] = z
        # pass 2: soft-max weighted normals
        sw = np.zeros(Ws, np.float32)
        ax = np.zeros(Ws, np.float32)
        ay_ = np.zeros(Ws, np.float32)
        az = np.zeros(Ws, np.float32)
        al = np.zeros(Ws, np.float32)
        at = np.zeros(Ws, np.float32)
        for i in range(n):
            cx, cy, cz, r, ay, clip = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4], S[i, 5]
            ry = r * ay * 1.12
            if yf < cy - ry or yf > cy + ry or yf > clip:
                continue
            rx = r * 1.12
            x0 = max(0, int(cx - rx))
            x1 = min(Ws, int(cx + rx) + 1)
            p1, p2, wob = S[i, 6], S[i, 7], S[i, 8]
            for x in range(x0, x1):
                xf = x + 0.5
                u = (xf - cx) / r
                v = (yf - cy) / (r * ay)
                th = math.atan2(v, u)
                rr = 1.0 + wob * (0.6 * math.sin(3.0 * th + p1) + 0.4 * math.sin(5.0 * th + p2))
                d2 = (u * u + v * v) / (rr * rr)
                if d2 >= 1.0:
                    continue
                s = math.sqrt(1.0 - d2)
                z = cz + r * s * dk
                w = math.exp((z - zmax[x]) / (kk * r))
                # edge falloff so a lobe's rim does not dominate the blend
                w *= min(1.0, s * 4.0 + 0.05)
                sw[x] += w
                ax[x] += w * u / rr
                ay_[x] += w * (-v / rr)
                az[x] += w * s
                al[x] += w * S[i, 9]
                at[x] += w * S[i, 10]
        for x in range(Ws):
            if sw[x] > 0:
                cov[y, x] = 1.0
                hz[y, x] = zmax[x]
                l = math.sqrt(ax[x] ** 2 + ay_[x] ** 2 + az[x] ** 2) + 1e-9
                nx[y, x] = ax[x] / l
                ny[y, x] = ay_[x] / l
                nz[y, x] = az[x] / l
                lv[y, x] = al[x] / sw[x]
                tn[y, x] = at[x] / sw[x]
    return cov, hz, nx, ny, nz, lv, tn


class Lobes:
    def __init__(self, rng):
        self.rows = []
        self.rng = rng

    def add(self, cx, cy, cz, r, ay=0.9, clip=1e9, level=0, wob=0.05, tone=0.0):
        self.rows.append([cx, cy, cz, r, ay, clip, self.rng.uniform(0, 6.28), self.rng.uniform(0, 6.28),
                          wob, level, tone])
        return len(self.rows) - 1

    def arr(self):
        return np.array(self.rows, np.float64)


def _tower(L, rng, axis_x, base_y, top_y, hw_fn, r_fn, clip, sun3, depth=0.7, zoff=0.0, density=1.0,
           top_round=True):
    """Level-0 lobes on the front surface of an elliptic tower envelope; returns their indices."""
    idx = []
    Hc = base_y - top_y
    v = 0.0
    while v <= 1.0:
        R = r_fn(v) * rng.uniform(0.85, 1.15)
        hw = hw_fn(v)
        y = base_y - v * Hc
        ax = axis_x(v)
        # arc length of the visible half ellipse ~ pi/2*(hw + d)
        dz = hw * depth
        arc = math.pi * 0.5 * (hw + dz)
        n = max(1, int(round(arc / (R * 1.05) * density)))
        for k in range(n):
            ph = -math.pi / 2 + math.pi * (k + 0.5) / n + rng.uniform(-0.25, 0.25) * math.pi / n
            Rk = R * rng.uniform(0.75, 1.2)
            push = 1.0 + (0.1 + 0.3 * rng.random()) * abs(math.sin(ph))
            x = ax + math.sin(ph) * (hw - Rk * 0.35) * push
            z = zoff + math.cos(ph) * dz
            yy = y + rng.uniform(-0.2, 0.2) * Rk
            # sun-side lobes slightly larger & pushed out: the lit face is the hero
            tone = rng.uniform(-1, 1)
            idx.append(L.add(x, yy, z, Rk, rng.uniform(0.84, 0.96), clip, 0, rng.uniform(0.02, 0.05), tone))
        v += R * rng.uniform(0.5, 0.68) / Hc
    if top_round:
        # crown: a few lobes capping the top
        R = r_fn(1.0)
        hw = hw_fn(1.0)
        for k in range(3):
            x = axis_x(1.0) + (k - 1) * hw * 0.55 + rng.uniform(-0.1, 0.1) * R
            idx.append(L.add(x, top_y - R * rng.uniform(0.1, 0.4), zoff + hw * depth * 0.4, R * rng.uniform(0.9, 1.2),
                             0.9, clip, 0, 0.04, rng.uniform(-1, 1)))
    return idx


def _children(L, rng, parents, count, ratio, sun3, clip, level, up_bias=0.55, sun_bias=0.6, min_face=-0.3,
              surf=0.88, pop=0.35):
    """Child lobes on the visible upper / sun-facing surfaces of the parents (3D directions)."""
    rows = L.rows
    out = []
    sx, sy, sz = sun3
    for pi in parents:
        cx, cy, cz, r = rows[pi][0], rows[pi][1], rows[pi][2], rows[pi][3]
        n = int(rng.integers(count[0], count[1] + 1))
        for _ in range(n):
            # random direction on the front hemisphere, biased up / toward the sun
            d = rng.normal(0, 1, 3)
            d[2] = abs(d[2]) * 0.8
            if rng.random() < up_bias:
                d = d + np.array([0.0, 1.6, 0.3])
            if rng.random() < sun_bias:
                d = d + 1.4 * np.array([sx, sy, sz])
            d = d / (np.linalg.norm(d) + 1e-9)
            face = d[0] * sx + d[1] * sy + d[2] * sz
            if face < min_face:
                continue
            rr = r * rng.uniform(ratio[0], ratio[1])
            x = cx + d[0] * r * surf
            y = cy - d[1] * r * surf * 0.92
            z = cz + d[2] * r * surf * 0.5 + rr * pop * 0.5
            if y > clip - rr * 0.2:
                continue
            tone = rows[pi][10] * 0.5 + rng.uniform(-0.5, 0.5)
            out.append(L.add(x, y, z, rr, rng.uniform(0.85, 0.97), clip, level, rng.uniform(0.03, 0.07), tone))
    return out


def build_massif(pw, ph, x0, base_y, top_y, sun3, seed=5):
    rng = np.random.default_rng(seed)
    L = Lobes(rng)
    Hc = base_y - top_y
    clip = base_y
    lean = 0.03 * pw

    def ax_main(v):
        return x0 + lean * v ** 1.2 + 0.012 * pw * math.sin(v * 5.0)

    ph1, ph2 = rng.uniform(0, 6.28, 2)

    def hw_main(v):
        return pw * float(np.interp(v, [0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0],
                                    [0.105, 0.098, 0.088, 0.082, 0.08, 0.078, 0.07])) * (
            1 + 0.1 * math.sin(v * 11 + ph1) + 0.06 * math.sin(v * 23 + ph2))

    def r_main(v):
        return pw * float(np.interp(v, [0, 0.3, 0.6, 1.0], [0.058, 0.046, 0.036, 0.028]))
    # shoulders first (they are behind the main column), then the column
    # left shoulder (lower, broad) and right shoulder (lower still, sun side -> brightly lit)
    shoulders = []
    for (dx, hv, hwf, rf, zo) in [(-0.12, 0.36, 0.06, 0.04, -0.06), (0.125, 0.24, 0.055, 0.036, -0.05),
                                  (-0.21, 0.15, 0.05, 0.032, -0.1), (0.21, 0.1, 0.045, 0.028, -0.1)]:
        tb = base_y - hv * Hc

        def axs(v, dx=dx):
            return x0 + dx * pw + 0.01 * pw * v

        def hws(v, hwf=hwf):
            return pw * hwf * float(np.interp(v, [0, 0.6, 1.0], [1.0, 0.85, 0.6]))

        def rs(v, rf=rf):
            return pw * rf * float(np.interp(v, [0, 1.0], [1.0, 0.7]))
        shoulders += _tower(L, rng, axs, base_y, tb, hws, rs, clip, sun3, depth=0.6, zoff=zo * pw)
    main = _tower(L, rng, ax_main, base_y, top_y, hw_main, r_main, clip, sun3, depth=0.75, zoff=0.0)
    lev0 = shoulders + main
    l1 = _children(L, rng, lev0, (2, 4), (0.36, 0.55), sun3, clip, 1, up_bias=0.6, sun_bias=0.55, min_face=-0.2,
                   pop=0.1)
    l2 = _children(L, rng, l1, (1, 3), (0.32, 0.5), sun3, clip, 2, up_bias=0.6, sun_bias=0.7, min_face=0.25,
                   pop=0.1)
    _children(L, rng, l2, (0, 2), (0.32, 0.5), sun3, clip, 3, up_bias=0.6, sun_bias=0.8, min_face=0.45, pop=0.1)
    return L, ax_main


def shade_massif(L, pw, ph, Wref, sun_xy, sun3, base_y, top_y, pal, ss=2, dk=0.8, kk=0.3, seed=3, hblur=0.003, hgain=1.0, own=0.5, cast_len=0.08, cast_slope=0.5,
                 extra_shadow=None):
    S = L.arr().copy()
    for k in (0, 1, 2, 3, 5):
        S[:, k] *= ss
    Hs, Ws = ph * ss, pw * ss
    cov, hz, nx, ny, nz, lv, tn = _smooth_union(Hs, Ws, S, dk, kk)

    def dn(a):
        return cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
    alpha = dn(cov)
    m = np.maximum(alpha, 1e-4)
    Nx, Ny, Nz = dn(nx * cov) / m, dn(ny * cov) / m, dn(nz * cov) / m
    Hh = dn(hz * cov) / m / ss
    Lv = dn(lv * cov) / m
    Tn = dn(tn * cov) / m
    del cov, hz, nx, ny, nz, lv, tn
    # painted form: normals from the (normalised-convolution) blurred height field -> soft cauliflower
    # modelling instead of per-sphere 'bubbles'; a little of the own-lobe normal keeps crisp small bumps
    hm0 = Hh * alpha
    s_h = hblur * Wref
    Hb = gblur(hm0, s_h) / (gblur(alpha, s_h) + 1e-4)
    hgx = cv2.Sobel(Hb, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    hgy = cv2.Sobel(Hb, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    hk = hgain
    mx_, my_, mz_ = -hgx * hk, hgy * hk, np.ones_like(hgx)
    ml = np.sqrt(mx_ ** 2 + my_ ** 2 + mz_ ** 2)
    Nx = own * Nx + (1 - own) * mx_ / ml
    Ny = own * Ny + (1 - own) * my_ / ml
    Nz = own * Nz + (1 - own) * mz_ / ml
    nl = np.sqrt(Nx ** 2 + Ny ** 2 + Nz ** 2) + 1e-6
    Nx, Ny, Nz = Nx / nl, Ny / nl, Nz / nl
    ys = np.arange(ph, dtype=np.float32)[:, None] * np.ones((1, pw), np.float32)
    xs = np.arange(pw, dtype=np.float32)[None, :] * np.ones((ph, 1), np.float32)
    lx, ly, lz = sun3
    # whole-massif form normal from a heavily blurred silhouette
    sg = 0.045 * Wref
    q_ = 8
    sm = cv2.resize(alpha, (pw // q_, ph // q_), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sg / q_)
    gx = cv2.resize(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * q_), (pw, ph), interpolation=cv2.INTER_CUBIC)
    gy = cv2.resize(cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * q_), (pw, ph), interpolation=cv2.INTER_CUBIC)
    kf = sg * 2.4
    fx, fy, fz = -gx * kf, gy * kf, np.ones_like(gx)
    fl = np.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    e_own = Nx * lx + Ny * ly + Nz * lz
    e_form = fx * lx + fy * ly + fz * lz
    e = 0.62 * e_own + 0.5 * e_form - 0.2
    # height: upper tower catches more sun; the lower core is shadowed by the mass above
    vh = np.clip((base_y - ys) / (base_y - top_y), 0, 1.2)
    e = e + 0.22 * (vh - 0.45)
    if extra_shadow is not None:
        e = e - extra_shadow
    # soft occlusion where lobes nest (height-field cavity at two scales)
    hm = Hh * alpha
    ao = 0.0
    for (s_, dv, wt) in ((0.006, 0.02, 0.6), (0.02, 0.06, 0.5)):
        s1 = s_ * Wref
        nb = fblur(hm, s1) / (fblur(alpha, s1) + 1e-4)
        ao = ao + wt * np.clip((nb - Hh) / (dv * Wref), 0, 1)
    ao = np.clip(ao, 0, 1) * (alpha > 0.01)
    ao = gblur(ao.astype(np.float32), 0.002 * Wref)
    e = e - 0.35 * ao
    # height-field self shadowing: nearer lobes cast crisp scalloped shadows onto the lobes behind/below
    # them (marched toward the sun in screen space) - the painted cauliflower structure
    cast = np.zeros_like(Hh)
    if cast_len > 0:
        dxs_, dys_ = lx, -ly
        dn2 = math.hypot(dxs_, dys_)
        dxs_, dys_ = dxs_ / dn2, dys_ / dn2
        slope = lz / dn2 * cast_slope
        Hf = np.where(alpha > 0.5, Hh, -1e4).astype(np.float32)
        step = max(1.0, 0.0012 * Wref)
        nst = int(cast_len * Wref / step)
        for k in range(1, nst + 1):
            t_ = k * step
            M = np.float32([[1, 0, -dxs_ * t_], [0, 1, -dys_ * t_]])
            Hq = cv2.warpAffine(Hf, M, (pw, ph), flags=cv2.INTER_LINEAR, borderValue=-1e4)
            soft = 0.5 + 0.03 * t_
            cast = np.maximum(cast, np.clip((Hq - Hh - t_ * slope) / soft, 0, 1))
        cast = gblur(cast * (alpha > 0.5), max(0.0006 * Wref, 0.5))
    # colour ramp
    P = pal
    k1 = ss_(-0.4, -0.05, e)           # core -> shadow
    k2 = ss_(-0.1, 0.0, e) * 0.5       # shadow -> half-tone (just under the terminator)
    k3 = ss_(0.0, 0.07, e) * (1 - cast)  # crisp painted terminator -> lit plane
    k4 = ss_(0.3, 0.75, e)             # lit -> hot
    sh = P['core'] + (P['sh'] - P['core']) * k1[..., None]
    # sky fill on upward-facing shadowed surfaces, warm ground bounce on downward faces of low lobes
    up = np.clip(Ny, 0, 1)[..., None]
    sh = sh + (P['sh_up'] - sh) * up * 0.55
    low = ss_(0.45, 0.0, vh)
    dn_ = (np.clip(-Ny, 0, 1) * low)[..., None]
    sh = sh + (P['bounce'] - sh) * dn_ * 0.75
    # tonal variety across the massif (painted, per-lobe)
    sh = sh * (1 + 0.035 * Tn)[..., None]
    col = sh + (P['half'] - sh) * k2[..., None]
    col = col + (P['lit'] - col) * k3[..., None]
    # modelling inside the lit plane: each lobe's lower/away side turns to a pale lilac half-light,
    # separating the cauliflower lobes without dark outlines
    lm = ss_(0.55, 0.0, e_own) * k3
    col = col + (P['lit_lo'] - col) * (lm * 0.9)[..., None]
    col = col + (P['hot'] - col) * (k4 * (1 - lm))[..., None]
    # occlusion colour: deep cool lavender in the nests (on top of whichever plane)
    col = col + (P['crev'] - col) * (np.clip(ao * 1.2, 0, 1) * (0.55 - 0.25 * k3))[..., None]
    # lobe separation inside the lit plane: fine cavity lines where a lobe tucks under its neighbour
    s0 = 0.003 * Wref
    nb0 = gblur(hm, s0) / (gblur(alpha, s0) + 1e-4)
    cav = np.clip((nb0 - Hh) / (0.008 * Wref), 0, 1) * (alpha > 0.5) * ss_(0.6, 0.1, e_own)
    col = col + (P['lit_crev'] - col) * (np.clip(cav * 1.5, 0, 1) * k3 * 0.85)[..., None]
    # thin-edge translucency (sky light through the cloud edges), all around the silhouette
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    edge = ss_(0.008 * Wref, 0.0, din) * (1 - k3)
    col = col + (P['edge'] - col) * (edge * 0.45)[..., None]
    # crisp silver rim on the sun-facing silhouette
    sx, sy = sun_xy
    dxs, dys = sx - xs, sy - ys
    dl = np.sqrt(dxs * dxs + dys * dys) + 1e-3
    ex, ey = dxs / dl, dys / dl
    cg = gblur(alpha, max(0.001 * Wref, 0.6))
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    face = np.clip((ogx * ex + ogy * ey) / ogl, 0, 1) * (ogl > 1e-3)
    face = ss_(0.1, 0.6, gblur(face.astype(np.float32), max(0.0015 * Wref, 0.7)))
    rw = 0.0013 * Wref
    rim = np.clip(ss_(rw * 2.0, rw * 0.5, din) * face, 0, 1)
    col = col + (P['rim'] - col) * rim[..., None]
    return np.dstack([col, alpha]).astype(np.float32), dict(e=e, alpha=alpha, ao=ao, k3=k3, cast=cast)


def streak_noise(pw, ph, seed, cx=6.0, cy=110.0, octaves=3, angle=0.0):
    """Fibrous (anisotropic) noise 0..1: long streaks along x."""
    rng = np.random.default_rng(seed)
    out = np.zeros((ph, pw), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gx_, gy_ = int(cx * 2 ** o) + 2, int(cy * 2 ** o) + 2
        g = rng.random((gy_, gx_)).astype(np.float32)
        out += amp * cv2.resize(g, (pw, ph), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    out /= tot
    if angle:
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (pw, ph), borderMode=cv2.BORDER_REFLECT)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def prof_noise(n, seed, freqs=((3, 1.0), (7, 0.5), (17, 0.3), (41, 0.15))):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    tot = 0
    for (f, a) in freqs:
        k = int(f * rng.uniform(0.8, 1.3)) + 3
        v = rng.uniform(-1, 1, k)
        xi = x * (k - 3) + rng.uniform(0, 1)
        i0 = np.floor(xi).astype(int)
        fr = xi - i0
        fr = fr * fr * (3 - 2 * fr)
        out += a * (v[i0] * (1 - fr) + v[np.minimum(i0 + 1, k - 1)] * fr)
        tot += a
    return (out / tot).astype(np.float32)


def paint_anvil(pw, ph, an, sun_xy, Wref, pal, seed=11):
    """Anvil: sheared ice shelf. Billowy lit top edge (crisp), soft gradient underside, fibrous streaks that
    trail downwind into cirrus and dissolve into the sky."""
    x0, x1 = an['x0'], an['x1']
    top0, bot0 = an['top'], an['bot']
    dome_x = an['dome_x']
    X = np.arange(pw, dtype=np.float32)
    u = (X - x0) / (x1 - x0)
    # top edge: broad dome over the column, gentle sag downwind; soft billows (ASYMMETRIC bumps)
    tn = prof_noise(pw, seed, ((4, 1.0), (11, 0.55), (27, 0.3), (63, 0.12)))
    top = (top0 + 0.035 * ph * (1 - np.exp(-((X - dome_x) / (0.17 * pw)) ** 2)) +
           0.02 * ph * ss_(0.5, 1.0, u) + tn * 0.012 * ph)
    brng = np.random.default_rng(seed + 3)
    bx = x0 + 0.02 * pw
    while bx < x0 + 0.72 * (x1 - x0):
        uu = (bx - x0) / (x1 - x0)
        bw = pw * brng.uniform(0.02, 0.05) * (1.1 - 0.5 * uu)
        bh = bw * brng.uniform(0.18, 0.3)
        t_ = (X - bx) / bw
        top = top - bh * np.sqrt(np.clip(1 - t_ * t_, 0, 1)) * (1 - 0.3 * np.clip(t_, 0, 1))
        bx += bw * brng.uniform(1.1, 1.7)
    # underside: rises downwind (shear), irregular broad sag
    bn = prof_noise(pw, seed + 1, ((3, 1.0), (8, 0.6), (19, 0.35), (47, 0.15)))
    bot = (bot0 - 0.045 * ph * ss_(0.3, 1.0, u) + bn * 0.02 * ph +
           0.02 * ph * np.exp(-((X - dome_x) / (0.1 * pw)) ** 2))
    # upwind rounded end
    cap = ss_(-0.01, 0.07, u)
    mid = 0.5 * (top + bot)
    half = 0.5 * (bot - top) * np.sqrt(np.clip(cap * (2 - cap), 0, 1))
    top_c = mid - half
    bot_c = mid + half
    Y = np.arange(ph, dtype=np.float32)[:, None]
    th = np.maximum(bot_c - top_c, 1.0)[None, :]
    v = (Y - top_c[None, :]) / th                                   # 0 top .. 1 underside
    uu = u[None, :] * np.ones((ph, 1), np.float32)
    fib = streak_noise(pw, ph, seed + 5, cx=4.0, cy=70.0, octaves=4, angle=1.5)
    fib2 = streak_noise(pw, ph, seed + 6, cx=10.0, cy=150.0, octaves=3, angle=0.8)
    # density: crisp at the lit top edge (1-2 px AA), SOFT underside (gradient fade), downwind dissolve
    aa = 1.0 / th
    d_top = ss_(-aa * 0.8, aa * 0.8, v)
    und_soft = 0.42 + 0.2 * ss_(0.3, 1.0, uu)
    d_bot = ss_(1.0 + 0.05, 1.0 - und_soft, v + (fib - 0.5) * 0.25)
    dens = d_top * d_bot * (uu > -0.05)
    tail = ss_(0.5, 1.05, uu)
    fibm = np.clip((fib * 0.6 + fib2 * 0.4) - 0.5 * tail + 0.2, 0, 1)
    dens = dens * (1 - tail + tail * ss_(0.05, 0.6, fibm))
    # cirrus streamers trailing beyond the tail (fade into the sky, tilted slightly up)
    cy_ = float(np.interp(x1, X, 0.5 * (top_c + bot_c)))
    Yc = Y - (cy_ - 0.02 * ph * (X[None, :] - x1) / (0.3 * pw))
    band = np.exp(-(Yc / (0.028 * ph)) ** 2)
    beyond = ss_(0.85, 1.05, uu) * ss_(1.6, 1.1, uu)
    wisp = np.clip(fib2 * 1.5 - 0.6, 0, 1) * band * beyond * 0.75
    alpha = np.clip(np.maximum(dens, wisp), 0, 1)
    # ---- shading
    lbn = prof_noise(pw, seed + 4, ((4, 1.0), (11, 0.6), (29, 0.3)))[None, :]
    lb = 0.55 + 0.12 * lbn + 0.05 * (fib - 0.5) + 0.1 * ss_(0.3, 1.0, uu)
    litk = ss_(lb + 0.12, lb - 0.08, v)
    litc = pal['lit'] + (pal['half'] - pal['lit']) * ss_(0.0, 1.0, v / np.maximum(lb, 0.1))[..., None] * 0.45
    litc = litc + (pal['hot'] - litc) * (ss_(0.3, 0.0, v) * ss_(0.75, 0.2, uu))[..., None] * 0.7
    und = pal['sh_up'] + (pal['sh'] - pal['sh_up']) * ss_(lb, 0.85, v)[..., None]
    ovc = np.exp(-((X[None, :] - dome_x) / (0.15 * pw)) ** 2) * ss_(lb, 1.0, v)
    und = und + (pal['core'] - und) * (ovc * 0.4)[..., None]
    # underside lit by the sky & the ground haze: lighter toward the soft bottom edge
    und = und + (pal['edge'] - und) * ss_(0.75, 1.05, v)[..., None] * 0.55
    col = und + (litc - und) * litk[..., None]
    fm = (fib - 0.5) * (0.02 + 0.12 * tail) + (fib2 - 0.5) * (0.015 + 0.06 * tail)
    col = col * (1 + fm)[..., None]
    col = col + (pal['tail'] - col) * (tail * 0.45)[..., None]
    col = np.where((wisp > dens)[..., None], pal['tail'] * 1.02, col)
    # silver rim along the lit top edge
    din = (Y - top_c[None, :])
    rim = ss_(0.003 * Wref, 0.0008 * Wref, din) * (din > -1) * ss_(0.95, 0.5, uu)
    col = col + (pal['rim'] - col) * (rim * 0.85)[..., None]
    return np.dstack([col, alpha]).astype(np.float32), dict(top=top_c, bot=bot_c, alpha=alpha)


PAL = dict(
    lit=hx('#f3e9d9'), lit_lo=hx('#c8c6e6'), lit_crev=hx('#aab0e0'), hot=np.array([0.99, 0.97, 0.92], np.float32), half=hx('#b8b8e2'),
    sh=hx('#939fdc'), sh_up=hx('#aab6ea'), core=hx('#7382c4'), crev=hx('#7280bc'),
    bounce=hx('#d6c4d0'), edge=hx('#c6d4f4'),
    rim=np.array([1.1, 1.08, 1.02], np.float32), tail=hx('#cfe0f6'),
)


def over(bot, top):
    ta = top[..., 3:4]
    ba = bot[..., 3:4] * (1 - ta)
    a = ta + ba
    rgb = (top[..., :3] * ta + bot[..., :3] * ba) / np.maximum(a, 1e-5)
    return np.concatenate([rgb, a], -1).astype(np.float32)


def base_wisps(pw, ph, Wref, x_lo, x_hi, base_y, seed, pal):
    """Torn, horizontally stretched scud under/along the flat base."""
    Y = np.arange(ph, dtype=np.float32)[:, None]
    X = np.arange(pw, dtype=np.float32)[None, :]
    fib = streak_noise(pw, ph, seed, cx=7.0, cy=80.0, octaves=4)
    fib2 = streak_noise(pw, ph, seed + 1, cx=18.0, cy=160.0, octaves=3)
    band = ss_(base_y - 0.03 * ph, base_y - 0.005 * ph, Y) * ss_(base_y + 0.04 * ph, base_y + 0.005 * ph, Y)
    span = ss_(x_lo, x_lo + 0.06 * pw, X) * ss_(x_hi, x_hi - 0.06 * pw, X)
    a = np.clip((fib * 0.7 + fib2 * 0.3 - 0.46) * 3.0, 0, 1) * band * span
    tone = ss_(base_y + 0.03 * ph, base_y - 0.02 * ph, Y)
    col = pal['sh'] + (pal['edge'] - pal['sh']) * (0.4 + 0.4 * (1 - tone))[..., None]
    col = np.broadcast_to(col, (ph, pw, 3)) * (1 + (fib2 - 0.5) * 0.08)[..., None]
    return np.dstack([col, a]).astype(np.float32)


def hero_cumulonimbus(pw, ph, Wref, x0, base_y, anvil, sun_xy, seed=5, sky=None, horizon_y=None):
    """Full hero thunderhead RGBA plate (straight alpha)."""
    sun3 = np.array([0.74, 0.5, 0.3])
    sun3 = sun3 / np.linalg.norm(sun3)
    top_y = anvil['bot'] + 0.02 * ph
    L, axis = build_massif(pw, ph, x0, base_y, top_y, sun3, seed=seed)
    arg, ainf = paint_anvil(pw, ph, anvil, sun_xy, Wref, PAL, seed=seed + 7)
    # cast shadow of the anvil on the upper column (soft band below the underside, shifted down-left)
    am = gblur(ainf['alpha'], 0.003 * Wref)
    acc = np.zeros_like(am)
    for k in range(1, 10):
        d = 0.006 * Wref * k
        M = np.float32([[1, 0, -0.9 * d], [0, 1, 0.7 * d]])
        acc = np.maximum(acc, cv2.warpAffine(am, M, (pw, ph)) * (1 - 0.09 * k))
    csh = fblur(acc * (1 - am), 0.008 * Wref) * 0.45
    rgba, inf = shade_massif(L, pw, ph, Wref, sun_xy, sun3, base_y, top_y, PAL, seed=seed, extra_shadow=csh)
    # overshooting dome: a few lit lobes bulging through the anvil top over the column
    rng = np.random.default_rng(seed + 40)
    O = Lobes(rng)
    dx = anvil['dome_x']
    ix = int(np.clip(dx, 0, pw - 1))
    ty = float(ainf['top'][ix])
    base_o = ty + 0.03 * ph
    for (ox, r, oy) in [(0.0, 0.05, 0.0), (-0.05, 0.036, 0.012), (0.045, 0.034, 0.014), (-0.09, 0.024, 0.024),
                        (0.085, 0.022, 0.026)]:
        O.add(dx + ox * pw, ty + oy * ph + r * pw * 0.35, 0.05 * pw, r * pw, 0.62, base_o, 0, 0.03, rng.uniform(-1, 1))
    l1 = _children(O, rng, list(range(len(O.rows))), (3, 5), (0.3, 0.45), sun3, base_o, 1, up_bias=0.8,
                   sun_bias=0.6, min_face=0.0)
    _children(O, rng, l1, (1, 3), (0.3, 0.45), sun3, base_o, 2, up_bias=0.8, sun_bias=0.7, min_face=0.2)
    orgba, _ = shade_massif(O, pw, ph, Wref, sun_xy, sun3, base_o, ty - 0.04 * ph, PAL, seed=seed + 1)
    # dome fades into the anvil body at its base
    Y = np.arange(ph, dtype=np.float32)[:, None]
    orgba[..., 3] *= ss_(base_o, base_o - 0.02 * ph, Y)
    out = over(rgba, arg)
    out = over(out, orgba)
    # aerial haze toward the base (lower tower melts into the horizon haze)
    if sky is not None:
        f = ss_(base_y - 0.35 * (base_y - top_y), base_y + 0.02 * ph, Y)[..., None] * 0.15
        out[..., :3] = out[..., :3] + (sky[..., :3] - out[..., :3]) * f
    wl = base_wisps(pw, ph, Wref, x0 - 0.33 * pw, x0 + 0.34 * pw, base_y, seed + 9, PAL)
    if sky is not None:
        wl[..., :3] = wl[..., :3] + (sky[..., :3] - wl[..., :3]) * 0.3
    out = over(out, wl)
    return out, dict(axis=axis, col_top=top_y)
