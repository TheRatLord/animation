"""s10 helper: painted towering cumulus (cauliflower lobe clusters) as an opaque RGBA plate.

Each tower is a hierarchy of 3D spheres (trunk lobes -> crown lobes -> small cauliflower heads).
They are rasterised in 2.5D: the front surface of every sphere is smooth-unioned (polynomial smax,
k ~ 0.35 r) into a 'bulge' map toward the camera, and the analytic sphere normals are blended with
the same smax weights, so lobes merge with soft fillets (no sphere-intersection creases) while a lobe
clearly in front of another keeps a crisp contour.  Shading is painterly: wrapped (half-Lambert) key
light from the sun side, soft lobe-on-lobe cast shadows (horizon-marched on the bulge map), cavity
AO, cool blue-lavender shadow side with warm bounce from the cloud sea below, a warm orange lit flank
and a silver-gold rim that bleeds inward along the sun-facing silhouette.  Opaque alpha, wispy torn
lower edges.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from lib import core as C


@njit(cache=True)
def _raster(hm, nm, cov, sp, ss):
    """sp rows: x, y (screen px, y down), z (toward camera), r, k.  All in output px (x ss)."""
    Hh, Ww = hm.shape
    for i in range(sp.shape[0]):
        cx = sp[i, 0] * ss
        cy = sp[i, 1] * ss
        cz = sp[i, 2] * ss
        r = sp[i, 3] * ss
        k = sp[i, 4] * ss
        x0 = max(int(cx - r) - 1, 0)
        x1 = min(int(cx + r) + 2, Ww)
        y0 = max(int(cy - r) - 1, 0)
        y1 = min(int(cy + r) + 2, Hh)
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / r
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                q = dx * dx + dy * dy
                if q >= 1.0:
                    continue
                ez = math.sqrt(1.0 - q)
                v = cz + r * ez
                # sphere normal (x right, y up, z toward camera)
                nx = dx
                ny = -dy
                nz = ez
                if cov[y, x] == 0:
                    hm[y, x] = v
                    nm[y, x, 0] = nx
                    nm[y, x, 1] = ny
                    nm[y, x, 2] = nz
                    cov[y, x] = 1
                    continue
                h0 = hm[y, x]
                d = abs(h0 - v)
                if d < k:
                    hh = (k - d) / k
                    nv = max(h0, v) + hh * hh * k * 0.25
                    w = min(max(0.5 + 0.5 * (v - h0) / k, 0.0), 1.0)
                elif v > h0:
                    nv = v
                    w = 1.0
                else:
                    continue
                hm[y, x] = nv
                a0 = nm[y, x, 0] * (1 - w) + nx * w
                a1 = nm[y, x, 1] * (1 - w) + ny * w
                a2 = nm[y, x, 2] * (1 - w) + nz * w
                n = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2) + 1e-9
                nm[y, x, 0] = a0 / n
                nm[y, x, 1] = a1 / n
                nm[y, x, 2] = a2 / n


@njit(cache=True, parallel=True)
def _shadow(hm, cov, lx, ly, lz, maxd, soft):
    """Lobe-on-lobe soft shadows: march from each pixel toward the light over the bulge map."""
    Hh, Ww = hm.shape
    out = np.ones((Hh, Ww), np.float32)
    lxy = math.sqrt(lx * lx + ly * ly) + 1e-9
    ux = lx / lxy
    uy = -ly / lxy
    slope = lz / lxy
    for y in prange(Hh):
        for x in range(Ww):
            if cov[y, x] == 0:
                continue
            h0 = hm[y, x]
            occ = 0.0
            s = 2.0
            while s < maxd:
                xx = int(x + ux * s)
                yy = int(y + uy * s)
                if xx < 0 or yy < 0 or xx >= Ww or yy >= Hh:
                    break
                if cov[yy, xx] != 0:
                    dh = hm[yy, xx] - (h0 + slope * s)
                    if dh > 0:
                        o = dh / (soft * s + 1.0)
                        if o > occ:
                            occ = o
                s += max(1.0, s * 0.06)
            out[y, x] = 1.0 - min(occ, 1.0)
    return out


def tower_spheres(rng, bx, by, height, width, lean=0.0, sun_side=1.0, crown=1.0, n_levels=9, detail=1.0):
    """Hierarchical lobes of a towering cumulus.  A designed envelope (broad massive body narrowing
    upward, a bulging cauliflower crown) is SHELLED with big lobes on its front half and sides, then
    medium and small heads grow outward/upward from them.  Returns (n, 5): x, y, z, r, level
    (screen px, y down; z toward the camera)."""
    S = []

    def hw_of(u):
        crown_b = 0.28 * crown * math.exp(-((u - 0.8) / 0.12) ** 2)
        return width * 0.5 * (1.0 - 0.42 * u + crown_b) * (1.0 - 0.6 * max(u - 0.93, 0.0) / 0.07)

    def cx_of(u):
        return bx + lean * height * u + 0.04 * width * math.sin(u * 5.0 + rng_ph)

    rng_ph = rng.uniform(0, 6.28)
    n0 = int(46 * detail)
    for i in range(n0):
        u = min(rng.uniform(0.0, 1.0) ** 0.8, 0.97)
        hw = hw_of(u)
        th = rng.uniform(-1.75, 1.75)
        x = cx_of(u) + hw * math.sin(th) * 0.72
        z = hw * math.cos(th) * 0.72
        y = by - height * u
        r = hw * rng.uniform(0.34, 0.5)
        S.append([x, y, z, r, 0, -1, math.sin(th), 0.0, math.cos(th)])
    # crown cap
    for i in range(int(9 * detail)):
        th = rng.uniform(-1.4, 1.4)
        ph = rng.uniform(0.0, 1.2)
        hw = hw_of(0.9)
        x = cx_of(0.95) + hw * 0.6 * math.sin(th) * math.cos(ph)
        y = by - height * 0.93 - hw * 0.35 * math.sin(ph)
        z = hw * 0.6 * math.cos(th) * math.cos(ph)
        r = hw * rng.uniform(0.3, 0.45)
        S.append([x, y, z, r, 0, -1, math.sin(th) * 0.6, 0.8, math.cos(th) * 0.6])
    level_specs = [(int(5 * detail), 0.3, 0.48, 0.45, 0.75), (int(4 * detail), 0.22, 0.36, 0.35, 0.65)]
    parents = list(range(len(S)))
    for lev, (nper, r0, r1, e0, e1) in enumerate(level_specs):
        newp = []
        for pi in parents:
            px, py, pz, pr = S[pi][0], S[pi][1], S[pi][2], S[pi][3]
            ox, oy, oz = S[pi][6], S[pi][7], S[pi][8]
            for _ in range(nper):
                a = rng.normal(0, 0.7, 3)
                a[0] += ox * 1.0 + 0.35 * sun_side
                a[1] = a[1] * 0.6 + oy + 0.9          # up (world y up)
                a[2] += oz * 1.0
                a /= np.linalg.norm(a) + 1e-9
                rc = pr * rng.uniform(r0, r1)
                d = pr - rc * rng.uniform(e0, e1)
                S.append([px + a[0] * d, py - a[1] * d, pz + a[2] * d, rc, lev + 1, pi, a[0], a[1], a[2]])
                newp.append(len(S) - 1)
        parents = newp
    A = np.array([s[:5] for s in S], np.float64)
    return A


def paint_towers(W, H, towers, light, pal, seed=3, ss=2):
    """towers: list of dicts(bx, by, height, width, lean, sun_side, crown, detail, haze, haze_col).
    light: (lx, ly, lz) direction TO the sun (x right, y up, z toward camera).
    Returns RGBA (H, W, 4) straight alpha plus the per-tower alpha list."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W, 4), np.float32)
    lx, ly, lz = light
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln
    P = {k: C.hex2rgb(v) for k, v in pal.items()}
    for T in towers:
        sp = tower_spheres(rng, T['bx'], T['by'], T['height'], T['width'], T.get('lean', 0.0),
                           T.get('sun_side', 1.0), T.get('crown', 1.0), T.get('levels', 9), T.get('detail', 1.0))
        kf = np.where(sp[:, 4] >= 2, 0.45, np.where(sp[:, 4] >= 1, 0.5, 0.35))[:, None]
        sp = np.concatenate([sp[:, :4], sp[:, 3:4] * kf], 1)   # smax k per lobe (small heads blend more)
        # boil / billow the silhouette: wobble the spheres slightly (static here; animated via warp)
        Hs, Ws = H * ss, W * ss
        hm = np.zeros((Hs, Ws), np.float32)
        nm = np.zeros((Hs, Ws, 3), np.float32)
        cov = np.zeros((Hs, Ws), np.uint8)
        # draw back-to-front order does not matter for smax, but sort by z for speed of early-outs
        _raster(hm, nm, cov, np.ascontiguousarray(sp), ss)
        a = cov.astype(np.float32)
        # downsample
        hm_d = cv2.resize(np.where(cov > 0, hm, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        a_d = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        hm_d = hm_d / np.maximum(a_d, 1e-4) / ss
        nm_d = cv2.resize(nm * a[..., None], (W, H), interpolation=cv2.INTER_AREA) / np.maximum(a_d, 1e-4)[..., None]
        # soften small-bump normals (keeps the cauliflower silhouette, avoids dark 'dimples')
        nsm = cv2.GaussianBlur(nm_d * a_d[..., None], (0, 0), 0.005 * W) / np.maximum(
            cv2.GaussianBlur(a_d, (0, 0), 0.005 * W), 1e-3)[..., None]
        nm_d = 0.45 * nm_d + 0.55 * nsm
        nn = np.sqrt((nm_d ** 2).sum(-1, keepdims=True)) + 1e-6
        nm_d = nm_d / nn
        cv_d = (a_d > 0.02).astype(np.uint8)
        # cast shadows between lobes
        hsm = cv2.GaussianBlur(hm_d, (0, 0), 0.006 * W)
        shd = _shadow(hsm.astype(np.float32), cv_d, lx, ly, lz, T['width'] * 0.9, 0.12)
        shd = cv2.GaussianBlur(shd, (0, 0), max(1.0, 0.003 * W))
        # cavity AO
        hb = cv2.GaussianBlur(hm_d * a_d, (0, 0), 0.012 * W) / np.maximum(cv2.GaussianBlur(a_d, (0, 0), 0.012 * W), 1e-3)
        cav = np.clip((hm_d - hb) / (0.02 * W), -1, 1)
        nx, ny, nz = nm_d[..., 0], nm_d[..., 1], nm_d[..., 2]
        ndl = nx * lx + ny * ly + nz * lz
        wrap = 0.4
        dif = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        dif = dif * dif * (3 - 2 * dif)
        key = dif * (0.2 + 0.8 * shd)
        # painterly brush break-up of the terminator
        bn = C.fbm(W, H, 60, 3, seed=seed + 11) - 0.5
        key = np.clip(key + bn * 0.18 * 4 * key * (1 - key), 0, 1)
        ao = np.clip(0.9 + cav * 0.3, 0.7, 1.05)
        # shadow side: cool blue-lavender, lighter up top (sky fill), warm bounce from the sea below
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        hrel = np.clip((T['by'] - yy * H) / T['height'], 0, 1)
        sh_col = C.lerp(P['shd_low'], P['shd_high'], (0.35 + 0.65 * hrel)[..., None])
        sh_col = C.lerp(sh_col, P['sky_fill'], (np.clip(ny, 0, 1) * 0.35)[..., None])
        sh_col = C.lerp(sh_col, P['bounce'], (np.clip(-ny, 0, 1) ** 1.2 * 0.6)[..., None])
        sh_col = sh_col * ao[..., None]
        # lit ramp
        stops = [(0.0, P['shd_high']), (0.22, P['mid']), (0.45, P['lit_low']), (0.7, P['lit']), (1.0, P['hi'])]
        lit = np.zeros((H, W, 3), np.float32)
        kk = key
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            tt = np.clip((kk - p0) / (p1 - p0), 0, 1)[..., None]
            seg = ((kk >= p0) & (kk <= p1))[..., None]
            lit = np.where(seg, C.lerp(c0, c1, tt), lit)
        mix = C.smoothstep(0.02, 0.3, key)[..., None]
        col = C.lerp(sh_col, lit * (0.9 + 0.1 * ao[..., None]), mix)
        # rim: silhouette toward the sun -> silver-gold, bleeds a few px inward
        R = max(2.0, 0.004 * W)
        ux, uy = lx / (math.hypot(lx, ly) + 1e-9), -ly / (math.hypot(lx, ly) + 1e-9)
        acc = np.zeros((H, W), np.float32)
        for k_, w_ in ((0.6, 0.4), (1.2, 0.35), (2.2, 0.25)):
            M = np.array([[1, 0, ux * R * k_], [0, 1, uy * R * k_]], np.float32)
            sh_a = cv2.warpAffine(a_d, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            acc += w_ * np.clip(a_d - sh_a, 0, 1)
        # also inner lobe contours facing the sun (front lobe edge over a farther lobe)
        hshift = cv2.warpAffine(hm_d, np.array([[1, 0, ux * R], [0, 1, uy * R]], np.float32), (W, H),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        inner = np.clip((hm_d - hshift) / (0.03 * W), 0, 1) * a_d
        rim = np.clip(acc * 1.6 + inner * 0.5, 0, 1) * np.clip(0.3 + 0.9 * (nx * lx + ny * ly) ** 1, 0, 1)
        rim = cv2.GaussianBlur(rim, (0, 0), 0.7)
        col = C.lerp(col, P['rim'], (rim * 0.6 * T.get('rim', 1.0))[..., None]) + P['rim'] * (rim * 0.45 * T.get('rim', 1.0))[..., None]
        # translucent glow just inside the sun-side silhouette
        glow = cv2.GaussianBlur(acc, (0, 0), R * 2.5)
        col = col + P['glow'] * (glow * 0.35 * T.get('rim', 1.0))[..., None]
        # aerial perspective: haze toward the horizon (lower part melts into the sea haze)
        hz = T.get('haze', 0.15) + 0.6 * (1 - C.smoothstep(0.0, 0.3, hrel)) ** 1.5
        col = C.lerp(col, np.asarray(T['haze_col'], np.float32), np.clip(hz, 0, 0.8)[..., None])
        # torn wispy lower / side edges: erode alpha with fibrous noise where the cloud is thin (low)
        fib = C.fbm(W, H, 90, 4, seed=seed + 5)
        fibh = cv2.resize(C.fbm(W // 4, H // 4, 8, 3, seed=seed + 6), (W, H))
        edge = 1.0 - C.smoothstep(0.55, 1.0, a_d)
        a_f = a_d * (1 - 0.6 * edge * (1 - fib))
        lowfade = C.smoothstep(0.0, 0.08, hrel)       # base dissolves into the sea
        a_f = a_f * (0.4 + 0.6 * lowfade + 0.3 * fibh * (1 - lowfade)).clip(0, 1)
        a_f = np.clip(a_f, 0, 1)
        # composite this tower over previous (towers listed back to front)
        o_a = out[..., 3:4]
        n_a = a_f[..., None]
        tot = n_a + o_a * (1 - n_a)
        out[..., :3] = (col * n_a + out[..., :3] * o_a * (1 - n_a)) / np.maximum(tot, 1e-5)
        out[..., 3:4] = tot
    return out.astype(np.float32)
