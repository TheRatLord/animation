"""Round-6 mountains for s07_comet_night: an eroded heightfield ray-cast from the lake, then PAINTED.

Why: the earlier 2-D range painters produced stamped 'curtain' snow with flat-bottomed vertical
streaks. Here the range is a real terrain (three ranges at increasing depth, ridged relief, droplet
hydraulic erosion -> dendritic gullies that follow the fall line, narrow downhill and branch around the
spurs). It is ray-cast column by column from the lake into three depth plates (near massif / far centre
peaks / faint back range) for parallax, and shaded as a painting:
  * light = the comet (upper right, slightly behind the range) -> broad stepped lit / shadow planes,
  * snow by altitude, held in the gullies (tapering tongues down the fall line), shed from steep ribs,
  * bare dark rock ribs between the gullies,
  * comet-lit cyan-white rim (2-3 px) on crests / faces turned toward the comet, hottest near it,
  * aerial perspective by distance (far ranges bluer, paler, softer), haze pooling at the feet.
"""
import math
import os

import numpy as np
import cv2
import numba

from lib import core as C

RES = 1.0 / 1150                 # heightmap cell in (u, v) units (~1.9 plate px)
U0, U1 = -0.56, 0.56
Z0, Z1 = 5.5, 50.0
V0, V1 = math.log(Z0), math.log(Z1)


# ----------------------------------------------------------------------------- noise
def _vnoise(w, h, cells_x, cells_y, seed):
    return C.value_noise(w, h, cells_x, cells_y, seed)


def ridged_mf(w, h, cells, octaves, seed, gain=0.5, lac=2.03):
    """Ridged multifractal (0..1): sharp crests, smooth valleys."""
    out = np.zeros((h, w), np.float32)
    wgt = np.ones((h, w), np.float32)
    amp, tot = 1.0, 0.0
    cx = cells
    for o in range(octaves):
        n = _vnoise(w, h, cx, cx * h / w, seed + 97 * o)
        r = (1 - np.abs(2 * n - 1)) ** 2
        r = r * wgt
        wgt = np.clip(r * 1.6, 0, 1)
        out += amp * r
        tot += amp
        amp *= gain
        cx *= lac
    return out / tot


# ----------------------------------------------------------------------------- erosion
@numba.njit(cache=True)
def _erode(hm, n_drops, seed, radius, inertia, cap_f, min_cap, dep_s, ero_s, evap, grav, life, hmin):
    Hz, Wx = hm.shape
    np.random.seed(seed)
    flux = np.zeros_like(hm)
    # brush
    offs = []
    wts = []
    tot = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = math.sqrt(dx * dx + dy * dy)
            if d <= radius:
                offs.append((dy, dx))
                wts.append(radius - d)
                tot += radius - d
    nb = len(offs)
    oy = np.empty(nb, np.int64)
    ox = np.empty(nb, np.int64)
    ow = np.empty(nb, np.float64)
    for i in range(nb):
        oy[i] = offs[i][0]
        ox[i] = offs[i][1]
        ow[i] = wts[i] / tot
    done = 0
    tries = 0
    while done < n_drops and tries < n_drops * 20:
        tries += 1
        px = np.random.random() * (Wx - 3) + 1
        py = np.random.random() * (Hz - 3) + 1
        if hm[int(py), int(px)] < hmin:
            continue
        done += 1
        dxv = 0.0
        dyv = 0.0
        speed = 1.0
        water = 1.0
        sed = 0.0
        for it in range(life):
            ix = int(px)
            iy = int(py)
            fx = px - ix
            fy = py - iy
            h00 = hm[iy, ix]
            h10 = hm[iy, ix + 1]
            h01 = hm[iy + 1, ix]
            h11 = hm[iy + 1, ix + 1]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            hh = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
            dxv = dxv * inertia - gx * (1 - inertia)
            dyv = dyv * inertia - gy * (1 - inertia)
            ln = math.sqrt(dxv * dxv + dyv * dyv)
            if ln < 1e-9:
                break
            dxv /= ln
            dyv /= ln
            flux[iy, ix] += water
            npx = px + dxv
            npy = py + dyv
            if npx < 1 or npx >= Wx - 2 or npy < 1 or npy >= Hz - 2:
                break
            jx = int(npx)
            jy = int(npy)
            gfx = npx - jx
            gfy = npy - jy
            nh = (hm[jy, jx] * (1 - gfx) * (1 - gfy) + hm[jy, jx + 1] * gfx * (1 - gfy) +
                  hm[jy + 1, jx] * (1 - gfx) * gfy + hm[jy + 1, jx + 1] * gfx * gfy)
            dh = nh - hh
            cap = max(-dh * speed * water * cap_f, min_cap)
            if sed > cap or dh > 0:
                amt = min(dh, sed) if dh > 0 else (sed - cap) * dep_s
                sed -= amt
                hm[iy, ix] += amt * (1 - fx) * (1 - fy)
                hm[iy, ix + 1] += amt * fx * (1 - fy)
                hm[iy + 1, ix] += amt * (1 - fx) * fy
                hm[iy + 1, ix + 1] += amt * fx * fy
            else:
                amt = min((cap - sed) * ero_s, -dh)
                for k in range(nb):
                    yy = iy + oy[k]
                    xx = ix + ox[k]
                    if yy < 0 or yy >= Hz or xx < 0 or xx >= Wx:
                        continue
                    e = amt * ow[k]
                    d = e if hm[yy, xx] >= e else hm[yy, xx]
                    hm[yy, xx] -= d
                    sed += d
            speed = math.sqrt(max(speed * speed + dh * grav, 0.0))
            water *= (1 - evap)
            px = npx
            py = npy
    return flux


# ----------------------------------------------------------------------------- ray cast
@numba.njit(cache=True)
def _edge_dist(Z, jump):
    """Rows since the last silhouette edge above (sky or a much farther surface)."""
    Hr, Wr = Z.shape
    out = np.full((Hr, Wr), 1e4, np.float32)
    for c in range(Wr):
        run = 1e4
        for r in range(Hr):
            z = Z[r, c]
            if not (z < 1e8):
                run = 1e4
                continue
            above = Z[r - 1, c] if r > 0 else 1e9
            if not (above < 1e8) or above > z * jump:
                run = 0.0
            else:
                run += 1.0
            out[r, c] = run
    return out


# ----------------------------------------------------------------------------- terrain build
def _env(pts, x):
    p = np.array(pts, np.float32)
    return np.interp(x, p[:, 0], p[:, 1]).astype(np.float32)


def build_heightmap(ranges, Wp, H, f, seed=5, cache_dir=None):
    """Terrain in (u, v) = (x/z, ln z) space with angular height a = h/z: a grid column IS a screen
    column and a cell has the same on-screen size at every depth (near and far ranges carry equal
    detail; aerial perspective does the softening). Locally this is an isotropic scaling of world space,
    so the erosion stays physically plausible. ranges: dict(zc, env=[(xs_frac, h_frac)], wz, back, bend).
    Returns grids in CELL units: a, flux, crest (angular crest height, world-free), range id."""
    import hashlib
    key = 'terrain3_%d_%s_%d_%d.npz' % (seed, hashlib.md5(repr(ranges).encode()).hexdigest()[:10], int(Wp), int(H))
    if cache_dir:
        pth = os.path.join(cache_dir, key)
        if os.path.exists(pth):
            d = np.load(pth)
            return d['hm'], d['flux'], d['crest'], d['rid']
    nu = int((U1 - U0) / RES)
    nv = int((V1 - V0) / RES)
    uu = U0 + (np.arange(nu, dtype=np.float32) + 0.5) * RES
    vv = V0 + (np.arange(nv, dtype=np.float32) + 0.5) * RES
    UU, VV = np.meshgrid(uu, vv)
    wa = (C.fbm(nu // 8, nv // 8, 8, 3, seed=seed + 1) - 0.5)
    wb = (C.fbm(nu // 8, nv // 8, 8, 3, seed=seed + 2) - 0.5)
    wa = cv2.resize(wa, (nu, nv), interpolation=cv2.INTER_CUBIC)
    wb = cv2.resize(wb, (nu, nv), interpolation=cv2.INTER_CUBIC)
    base = np.zeros((nv, nu), np.float32)
    crest = np.zeros((nv, nu), np.float32)
    rid = np.zeros((nv, nu), np.float32)
    kx = f / Wp
    for ri, r in enumerate(ranges):
        vc = math.log(r['zc'])
        c = _env(r['env'], (0.5 + UU * kx).ravel()).reshape(UU.shape) * H / f * r.get('scale', 1.0)
        wz = np.maximum(c * r.get('wz', 2.0), 0.01)
        wbk = np.maximum(c * r.get('back', 1.6), 0.01)
        dv = VV - vc + wb * r.get('bend', 1.5) * 0.12
        u_ = np.where(dv < 0, -dv / wz, dv / wbk)
        prof = np.clip(1 - u_, 0, 1)
        prof = prof ** 1.25 * (1 - 0.25 * (1 - prof) ** 6)
        hr = c * prof * r.get('ridge', 0.8)
        # a chain of peaks (cones) along the range: each erodes into a radial fan of spurs and gullies,
        # so the fall lines run diagonally down the flanks instead of straight down one flat wall
        rng = np.random.default_rng(seed + 100 + ri)
        env_u = np.array(r['env'], np.float32)
        ua = (env_u[0, 0] - 0.5) / kx
        ub = (env_u[-1, 0] - 0.5) / kx
        uq = max(ua, U0 - 0.05) + rng.uniform(0, 0.03)
        while uq < min(ub, U1 + 0.05):
            cp = float(np.interp(0.5 + uq * kx, env_u[:, 0], env_u[:, 1])) * H / f * r.get('scale', 1.0)
            if cp > 0.004:
                ch_ = cp * rng.uniform(0.72, 0.95)
                vq = vc + rng.normal(0, 0.25) * cp * r.get('wz', 2.0)
                rad = ch_ * r.get('cone', 1.5) * rng.uniform(0.85, 1.2)
                sel = (np.abs(uu - uq) < rad * 1.2)
                if sel.any():
                    j0, j1 = np.nonzero(sel)[0][[0, -1]]
                    du = UU[:, j0:j1 + 1] - uq
                    dvv = (VV[:, j0:j1 + 1] - vq) * np.where(VV[:, j0:j1 + 1] < vq, 1.0 / r.get('wz', 2.0),
                                                             1.0 / r.get('back', 1.6)) * 1.5
                    dd = np.sqrt(du * du + dvv * dvv) / rad
                    cone = ch_ * np.clip(1 - dd, 0, 1) ** 1.15
                    hr[:, j0:j1 + 1] = np.maximum(hr[:, j0:j1 + 1], cone)
            uq += rng.uniform(0.035, 0.075) * (0.5 + 0.5 * min(cp / 0.1, 1.5))
        take = hr > base
        crest = np.where(take, c, crest)
        rid = np.where(take, ri, rid)
        base = np.maximum(base, hr)
    rm = ridged_mf(nu, nv, (U1 - U0) / 0.16, 4, seed + 11, gain=0.45)
    rm2 = ridged_mf(nu, nv, (U1 - U0) / 0.05, 2, seed + 13)
    hmask = np.clip(base / (crest + 1e-6), 0, 1)
    h = base * (0.62 + 0.72 * rm + 0.07 * rm2) + 0.002 * hmask
    hm = (h / RES).astype(np.float32)
    sc = float(hm.max())
    hn = (hm / sc).astype(np.float64)
    n_drops = int(1.4 * nu * nv)
    flux = _erode(hn, n_drops, seed + 21, 2, 0.1, 8.0, 0.0005, 0.25, 0.6, 0.015, 4.0, 90, 0.004)
    hm = (hn * sc).astype(np.float32)
    # erosion notches every receding crest into a saw-tooth when seen edge-on: round them off
    hm = cv2.GaussianBlur(hm, (0, 0), 1.1)
    flux = flux.astype(np.float32)
    crest = (crest / RES).astype(np.float32)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(os.path.join(cache_dir, key), hm=hm, flux=flux, crest=crest, rid=rid)
    return hm, flux, crest, rid


@numba.njit(cache=True)
def _cast_uv(hm, u0, v0, res, Wr, Hr, cxp, yH, f, va, vb, dv, y_off, outU, outV):
    Hv, Wu = hm.shape
    for c in range(Wr):
        u = (c + 0.5 - cxp) / f
        gu = (u - u0) / res
        if gu < 0 or gu >= Wu - 1:
            continue
        iu = int(gu)
        fu = gu - iu
        ymin = float(Hr)
        v = va
        pv = va
        py_ = 1e9
        while v < vb:
            gv = (v - v0) / res
            if gv >= Hv - 1:
                break
            iv = int(gv)
            fv = gv - iv
            a = ((hm[iv, iu] * (1 - fu) + hm[iv, iu + 1] * fu) * (1 - fv) +
                 (hm[iv + 1, iu] * (1 - fu) + hm[iv + 1, iu + 1] * fu) * fv) * res
            y = yH - a * f - y_off
            if py_ > 1e8:
                py_ = y
            if y < ymin:
                r0 = int(math.ceil(y - 0.5))
                if r0 < 0:
                    r0 = 0
                r1 = int(math.ceil(ymin - 0.5))
                if r1 > Hr:
                    r1 = Hr
                for r in range(r0, r1):
                    t = 1.0
                    if py_ - y > 1e-6:
                        t = (py_ - (r + 0.5)) / (py_ - y)
                        if t < 0:
                            t = 0.0
                        if t > 1:
                            t = 1.0
                    outU[r, c] = u
                    outV[r, c] = pv + (v - pv) * t
                ymin = y
                if ymin <= 0:
                    break
            py_ = y
            pv = v
            v += dv


def cast(hm, Wp, Hl_rows, y_off, yH, f, za, zb, ss=2):
    """Returns per-pixel (U, Z) at ss x plate resolution; Z = 1e9 where nothing was hit."""
    Wr, Hr = Wp * ss, Hl_rows * ss
    outU = np.zeros((Hr, Wr), np.float32)
    outV = np.full((Hr, Wr), 1e9, np.float32)
    _cast_uv(hm, U0, V0, RES, Wr, Hr, Wp * ss / 2.0, yH * ss, f * ss, math.log(za), math.log(zb), RES * 0.25,
             y_off * ss, outU, outV)
    Z = np.where(outV < 1e8, np.exp(np.minimum(outV, 10)), 1e9).astype(np.float32)
    return outU, Z


def _sample(grid, U, Z):
    mx = ((U - U0) / RES - 0.5).astype(np.float32)
    mz = ((np.log(np.maximum(Z, 1e-3)) - V0) / RES - 0.5).astype(np.float32)
    return cv2.remap(grid, mx, mz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def shade(X, Z, fields, pal, light_x, light_y, Wp, H, f, s, yH, y_off, ss, aerial_col, fog_d=30.0,
          rim_col=(0.72, 0.93, 1.0), rim_amt=1.0, snow_lo=0.42, soft=0.0, tl=0.28, seed=0):
    """Paint one depth plate. Returns straight RGBA at plate resolution (area-downsampled from ss)."""
    hm, nxg, nyg, nzg, fl, cv_, crest, n1, n2, ffg, n3g, lfg = fields
    valid = Z < 1e8
    Zs = np.where(valid, Z, 30.0).astype(np.float32)
    Xs = np.where(valid, X, 0.0).astype(np.float32)
    h = _sample(hm, Xs, Zs) * RES * Zs          # world height
    nx = _sample(nxg, Xs, Zs)
    ny = _sample(nyg, Xs, Zs)
    nz = _sample(nzg, Xs, Zs)
    flx = _sample(fl, Xs, Zs)
    cur = _sample(cv_, Xs, Zs)
    cr = np.maximum(_sample(crest, Xs, Zs) * RES * Zs, 0.05)
    na = _sample(n1, Xs, Zs)
    nb = _sample(n2, Xs, Zs)
    Hr, Wr = X.shape
    cols = (np.arange(Wr, dtype=np.float32) + 0.5) / ss
    xs_scr = np.broadcast_to(cols[None, :], (Hr, Wr))
    rows = (np.arange(Hr, dtype=np.float32) + 0.5) / ss + y_off
    ys_scr = np.broadcast_to(rows[:, None], (Hr, Wr))

    # ---- light: from the comet -> horizontal component points at it, up, a little behind the range
    lxs = np.clip((light_x - xs_scr) / (0.28 * Wp), -1, 1)
    Lx, Ly, Lz = 0.9 * lxs, 0.35, 0.3
    ln = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    dif = (nx * Lx + ny * Ly + nz * Lz) / ln
    # fine fluting: the small rills / ribs down the fall line catch and lose the light
    ffn = _sample(ffg, Xs, Zs)
    nf = _sample(n3g, Xs, Zs)
    flute = np.clip(ffn, -1.2, 1.2) * np.sign(lxs + 1e-3) * -0.5
    dif = dif + 0.16 * flute * _ss(0.05, 0.3, 1 - ny)
    # painted: broad value steps with crisp transitions
    v = 0.5 * _ss(0.33, 0.37, dif) + 0.5 * _ss(0.6, 0.64, dif)
    v = 0.8 * v + 0.2 * np.clip(dif * 0.9 + 0.2, 0, 1)
    prox = np.exp(-np.abs(xs_scr - light_x) / (0.45 * Wp))

    # ---- snow: altitude (relative to the range crest), held in gullies, shed from steep ribs
    hrel = h / cr
    gul = np.clip(0.6 * flx + 0.5 * cur, -1.5, 1.5)             # >0 gully / couloir, <0 rib
    # every peak gets its own gully character: some deeply fluted, some smooth wind-packed snow fields
    # with only a few couloirs (no single repeated stroke pattern across the range)
    xq = xs_scr / Wp
    pk_var = 0.5 + 0.5 * np.sin(xq * 23.0 + 1.3) * np.cos(xq * 9.0 + 0.4)
    gmod = 0.35 + 1.15 * _ss(0.2, 0.8, pk_var + 0.25 * (na - 0.5))
    gul = gul * gmod + 0.35 * (nb - 0.5) * (1 - gmod * 0.6)
    fln = np.clip(gul, 0, 1.2)
    steep = 1 - ny                                                # 0 flat .. 1 vertical
    edge_w = 0.03 + 0.05 * soft
    sline = snow_lo + 0.14 * (na - 0.5)
    sn = (hrel - sline) + 0.3 * gul * _ss(0.1, 0.4, hrel) - 0.45 * _ss(0.3, 0.55, steep) * _ss(0.3, -0.2, gul) * _ss(0.92, 0.7, hrel) \
        + 0.05 * (nb - 0.5)
    snow = _ss(-edge_w, edge_w, sn)
    if os.environ.get('TDBG'):
        m = valid
        for nm, a_ in (('hrel', hrel), ('fln', fln), ('steep', steep), ('cur', cur), ('dif', dif), ('snow', snow), ('gul', gul)):
            print(nm, np.percentile(a_[m], [5, 25, 50, 75, 95]).round(3))
            cv2.imwrite(os.environ['TDBG'] + f'_{nm}.png', cv2.resize(np.clip(a_ * m * 255 if nm != 'gul' else (a_ + 1) * 120 * m, 0, 255).astype(np.uint8), None, fx=0.5, fy=0.5))
    # bare rock ribs on the convex spur crests below the summits (never in the gullies)
    rib = _ss(0.15, 0.6, -gul) * _ss(0.95, 0.65, hrel) * _ss(0.12, 0.3, steep)
    snow = snow * (1 - 0.85 * rib)
    if os.environ.get('TDBG'):
        cv2.imwrite(os.environ['TDBG'] + '_rib.png', cv2.resize((rib * 255).astype(np.uint8), None, fx=0.5, fy=0.5))
    # painted structure: dark rock outcrops breaking through the snow along the fine ribs (crisp, elongated
    # down the fall line), wind-scoured bare patches on the steep upper faces
    oc = _ss(0.32, 0.46, -ffn * 0.8 + 0.55 * (nf - 0.5) + 0.25 * _ss(0.2, 0.6, steep)) *         _ss(0.08, 0.3, steep) * _ss(1.02, 0.85, hrel) * _ss(-0.1, 0.25, -gul + 0.3)
    snow = snow * (1 - 0.92 * oc)
    forest = _ss(tl + 0.03, tl - 0.03, hrel + 0.05 * (nb - 0.5) - 0.06 * fln) * (1 - snow)
    # forest band broken by snowy clearings / avalanche paths running down the gullies
    clear = _ss(0.35, 0.6, fln + 0.4 * (nf - 0.5)) * _ss(tl - 0.12, tl - 0.02, hrel)
    forest = forest * (1 - 0.85 * clear)
    snow = np.maximum(snow, clear * _ss(tl - 0.12, tl - 0.02, hrel) * 0.8)

    P = {k: np.asarray(c, np.float32) for k, c in pal.items()}
    rock = P['rock_sh'] + (P['rock_lit'] - P['rock_sh']) * v[..., None]
    rock = rock * (1 + 0.1 * (nb - 0.5)[..., None])
    # rock strata (slanted bedding lines) + crisp lit facets on the planes turned toward the comet
    strat = np.abs(np.sin(np.pi * (hrel * 16.0 + 4.0 * (na - 0.5) + 2.0 * (nf - 0.5) + 0.02 * xs_scr / s)))
    rock = rock * (1 - 0.3 * _ss(0.86, 0.96, strat)[..., None])
    facet = _ss(0.54, 0.58, nf + 0.25 * (dif - 0.5)) * _ss(0.35, 0.65, dif)
    rock = rock + (P['rock_lit'] * 1.35 - rock) * (0.55 * facet)[..., None]
    sv = np.clip(0.12 + 0.88 * v, 0, 1)
    snowc = P['snow_sh'] + (P['snow_lit'] - P['snow_sh']) * sv[..., None]
    hi = _ss(0.55, 0.85, dif) * prox
    snowc = snowc + (P['snow_hi'] - P['snow_lit']) * hi[..., None]
    # snow in the couloirs sits in shadow a touch deeper (reads the gully form)
    snowc = snowc * (1 - 0.12 * np.clip(fln, 0, 1)[..., None] * (1 - v[..., None]))
    fc = P['forest_sh'] + (P['forest_lit'] - P['forest_sh']) * v[..., None]
    fc = fc * (1 + 0.12 * (na - 0.5)[..., None])
    img = rock * (1 - snow[..., None]) + snowc * snow[..., None]
    img = img * (1 - forest[..., None]) + fc * forest[..., None]
    img = img * (0.85 + 0.25 * prox[..., None])
    # painted linework: thin lit strokes along the spur crests (sharp convex), thin dark strokes in the
    # gully floors -- the way a background painter traces a range's structure
    lc = _sample(lfg, Xs, Zs)
    rl = _ss(0.45, 0.85, -lc) * _ss(0.05, 0.25, 1 - ny) * (1 - forest) * _ss(0.25, 0.6, dif + 0.15)
    gl_ = _ss(0.5, 0.95, lc) * _ss(0.05, 0.25, 1 - ny) * (1 - 0.5 * forest)
    img = img + rl[..., None] * (P['snow_hi'] - img) * 0.4
    img = img * (1 - 0.28 * gl_[..., None])
    # brush texture: low-amplitude streaky value variation down the fall line
    img = img * (1 + 0.07 * (nf - 0.5)[..., None] + 0.05 * (na - 0.5)[..., None])

    # ---- aerial perspective + haze pooling at the feet
    fog = 1 - np.exp(-np.clip(Zs - 6.0, 0, None) / fog_d)
    low = np.exp(-np.clip(h, 0, None) / (0.1 * cr + 0.1)) * 0.55
    ac = np.asarray(aerial_col, np.float32)
    k = np.clip(fog + low * (1 - fog), 0, 0.92)
    img = img * (1 - k[..., None]) + ac * k[..., None]

    # ---- comet rim on crests / faces turned toward the comet
    ed = _edge_dist(np.where(valid, Z, 1e9).astype(np.float32), 1.08)
    rw = (1.5 * s + 0.2) * ss
    facing = np.clip(0.3 + nx * lxs * 1.0 + ny * 0.35, 0, 1)
    near_c = 0.25 + 0.75 * np.exp(-np.abs(xs_scr - light_x) / (0.32 * Wp))
    rim = (np.exp(-ed / rw) * 0.85 + np.exp(-ed / (rw * 3.5)) * 0.25) * facing ** 1.2 * near_c
    rim = rim * (0.6 + 0.4 * snow) * (1 - forest) * (1 - 0.4 * fog)
    img = img + rim[..., None] * np.asarray(rim_col, np.float32) * rim_amt
    # faces turned toward the comet carry a cool cyan cast (warm-cool split against the shadow planes)
    face = _ss(0.45, 0.8, dif) * near_c * (0.4 + 0.6 * snow) * (1 - 0.6 * fog)
    img = img + face[..., None] * np.array([0.02, 0.07, 0.1], np.float32) * rim_amt
    # lit snow faces toward the comet get a soft sheen too
    img = img + (hi * snow * facing)[..., None] * np.asarray(rim_col, np.float32) * 0.08 * rim_amt

    a = valid.astype(np.float32)
    rgba = np.dstack([img * a[..., None], a]).astype(np.float32)
    Wp_, Hr1 = Wr // ss, Hr // ss
    out = cv2.resize(rgba, (Wp_, Hr1), interpolation=cv2.INTER_AREA)
    if soft > 0:
        out = cv2.GaussianBlur(out, (0, 0), soft * s + 0.01)
    rgb = out[..., :3] / np.maximum(out[..., 3:4], 1e-5)
    return np.dstack([rgb, out[..., 3]]).astype(np.float32)


def fields_from(hm, flux, crest, seed=3):
    a = hm * RES
    hs = cv2.GaussianBlur(a, (0, 0), 0.9)
    gx = np.gradient(hs, axis=1) / RES                    # dh/dx = da/du
    gz = np.gradient(hs, axis=0) / RES + hs               # dh/dz = da/dv + a
    n = 1 / np.sqrt(gx * gx + gz * gz + 1)
    nxg, nyg, nzg = -gx * n, n, -gz * n
    hw = a
    # curvature (laplacian of a smoothed surface): >0 concave (gully), <0 convex (rib)
    hb = cv2.GaussianBlur(hw, (0, 0), 3.0)
    lap = cv2.Laplacian(hb, cv2.CV_32F)
    cv_ = np.clip(lap / (np.percentile(np.abs(lap), 95) + 1e-6), -1.5, 1.5)
    lf = np.log1p(cv2.GaussianBlur(flux, (0, 0), 1.0))
    # channel-ness: flux relative to its neighbourhood (>0 gully floor, <0 rib crest)
    fl = lf - cv2.GaussianBlur(lf, (0, 0), 6.0)
    fl = np.clip(fl / (np.percentile(np.abs(fl), 95) + 1e-6), -1.5, 1.5)
    nzs, nxs = hm.shape
    n1 = C.fbm(nxs // 4, nzs // 4, 40, 4, seed=seed)
    n1 = cv2.resize(n1, (nxs, nzs), interpolation=cv2.INTER_CUBIC)
    n2 = C.fbm(nxs // 2, nzs // 2, 160, 3, seed=seed + 1)
    n2 = cv2.resize(n2, (nxs, nzs), interpolation=cv2.INTER_CUBIC)
    # fine rills / ribs (small-scale channel-ness) and fine noise for outcrops, strata and facets
    ff = lf - cv2.GaussianBlur(lf, (0, 0), 1.8)
    ff = np.clip(ff / (np.percentile(np.abs(ff), 95) + 1e-6), -1.5, 1.5)
    n3 = C.fbm(nxs, nzs, 520, 3, seed=seed + 2)
    # fine curvature (thin ridge-spur / gully lines, drawn as painted linework)
    l2 = cv2.Laplacian(cv2.GaussianBlur(hw, (0, 0), 1.3), cv2.CV_32F)
    lf2 = np.clip(l2 / (np.percentile(np.abs(l2), 96) + 1e-9), -1.5, 1.5)
    return (hm.astype(np.float32), nxg.astype(np.float32), nyg.astype(np.float32), nzg.astype(np.float32),
            fl.astype(np.float32), cv_.astype(np.float32), crest.astype(np.float32), n1, n2,
            ff.astype(np.float32), n3.astype(np.float32), lf2.astype(np.float32))


# ----------------------------------------------------------------------------- scene-level entry point
def _hex(hh):
    return C.hex2rgb(hh)


RANGES = [
    # near left massif (darkest, crispest)
    dict(zc=10.0, env=[(-0.2, 0.24), (0.0, 0.25), (0.07, 0.3), (0.16, 0.26), (0.27, 0.18), (0.38, 0.1),
                       (0.46, 0.04), (0.52, 0.0)], wz=1.3, back=1.6, bend=1.5, scale=1.3),
    # near right ridge (toward the Milky Way)
    dict(zc=12.0, env=[(0.55, 0.0), (0.64, 0.05), (0.72, 0.1), (0.8, 0.16), (0.9, 0.225), (1.0, 0.215),
                       (1.2, 0.2)], wz=1.4, back=1.6, bend=1.5, scale=1.2),
    # far centre peaks seen through the valley
    dict(zc=22.0, env=[(0.1, 0.08), (0.25, 0.1), (0.35, 0.14), (0.42, 0.17), (0.5, 0.135), (0.57, 0.15),
                       (0.63, 0.185), (0.7, 0.15), (0.8, 0.13), (0.95, 0.1)], wz=1.8, back=1.5, bend=2.0,
         scale=1.12),
    # faint back range
    dict(zc=39.0, env=[(0.2, 0.12), (0.33, 0.15), (0.45, 0.205), (0.53, 0.17), (0.6, 0.22), (0.7, 0.18),
                       (0.85, 0.15), (1.0, 0.12)], wz=1.7, back=1.3, bend=2.5, scale=1.12),
]

PAL_NEAR = dict(rock_sh=_hex('#0a0f30'), rock_lit=_hex('#2c407a'), snow_sh=_hex('#28306e'), snow_lit=_hex('#7896d4'),
                snow_hi=_hex('#d2e4ff'), forest_sh=_hex('#0a1640'), forest_lit=_hex('#172e66'))
PAL_MID = dict(rock_sh=_hex('#1a2560'), rock_lit=_hex('#34508e'), snow_sh=_hex('#3a4892'), snow_lit=_hex('#86a4da'),
               snow_hi=_hex('#c4d8f6'), forest_sh=_hex('#1a2e62'), forest_lit=_hex('#27417c'))
PAL_FAR = dict(rock_sh=_hex('#33407e'), rock_lit=_hex('#4a5c9c'), snow_sh=_hex('#5460a4'), snow_lit=_hex('#8a9ad2'),
               snow_hi=_hex('#9cb8e6'), forest_sh=_hex('#2a4580'), forest_lit=_hex('#36568f'))


def range_plates(Wp, H, W, y_h, b0, Hl, light, cache_dir=None, ss=2):
    """Three RGBA plates (rows b0..Hl of the plate) + parallax factors, back to front."""
    f = float(Wp)
    s = W / 1920.0
    yH = y_h - 0.004 * H
    hm, flux, crest, rid = build_heightmap(RANGES, Wp, H, f, seed=5, cache_dir=cache_dir)
    fld = fields_from(hm, flux, crest)
    Hr = Hl - b0
    spec = [  # za, zb, palette, aerial colour, fog distance, softness, snowline, rim, parallax
        (29.0, 50.0, PAL_FAR, _hex('#6a6fb4'), 18.0, 0.9, 0.5, 0.8, 0.08, 0.3),
        (15.5, 29.0, PAL_MID, _hex('#5a72b6'), 26.0, 0.45, 0.45, 1.3, 0.16, 0.7),
        (5.5, 15.5, PAL_NEAR, _hex('#3a66aa'), 60.0, 0.0, 0.42, 2.2, 0.3, 0.9),
    ]
    out = []
    for (za, zb, pal, ac, fogd, soft, sl, rim, par, rim_rim), wr in zip(spec, (0.55, 0.5, 0.42)):
        X, Z = cast(hm, Wp, Hr, b0, yH, f, za, zb, ss)
        rgba = shade(X, Z, fld, pal, light[0], light[1], Wp, H, f, s, yH, b0, ss, ac, fog_d=fogd, soft=soft,
                     snow_lo=sl, rim_amt=rim)
        rgba = silhouette_rim(rgba, light[0], Wp, s, amt=rim_rim, reach=0.42)
        rgba = warm_rim(rgba, 0.3 * Wp, Wp, s, amt=wr)
        out.append((rgba, par))
    return out


def silhouette_rim(rgba, light_x, Wp, s, amt=0.85, reach=0.4, col=(0.8, 0.96, 1.06)):
    """Crisp 2-3 px comet-lit cyan-white rim along the plate's sky silhouette where it faces the comet
    (above, toward light_x), brightest near the comet axis and fading outward."""
    a = rgba[..., 3]
    Hh, Ww = a.shape
    inside = (a > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32) + (a - 0.5).clip(-0.5, 0.5) * 0
    gm = cv2.GaussianBlur(a, (0, 0), 2.0 * s + 0.6)
    gx, gy = np.gradient(gm, axis=1), np.gradient(gm, axis=0)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    xs = np.arange(Ww, dtype=np.float32)[None, :]
    lx = np.clip((light_x - xs) / (0.25 * Wp), -1, 1) * 0.8
    ly = -1.0
    ln = math.sqrt(1.0) + 0 * lx
    ln = np.sqrt(lx * lx + ly * ly)
    # inward normal = +grad(a); outward = -grad -> facing = dot(-grad, light dir)
    facing = np.clip((-gx * lx - gy * ly) / (gn * ln), 0, 1)
    facing = _ss(0.25, 0.85, facing)
    near = 0.42 + 0.58 * np.exp(-np.abs(xs - light_x) / (reach * Wp))
    rw = 1.25 * s + 0.35
    core = _ss(rw * 2.0, rw * 0.9, dist) * (dist > 0)
    r = (core * 0.95 + np.exp(-dist / (rw * 3.0)) * 0.12 * (dist > 0)) * facing * near * amt
    r = np.clip(r, 0, 0.92)
    out = rgba.copy()
    c = np.asarray(col, np.float32)
    out[..., :3] = rgba[..., :3] * (1 - r[..., None]) + c * r[..., None]
    return out


def warm_rim(rgba, x_warm, Wp, s, amt=0.5, col=(1.0, 0.66, 0.48)):
    """Thin warm twilight rim on silhouette edges turned toward the low horizon glow (left side of
    each peak) -- the warm half of the warm/cool split against the cyan comet rim."""
    a = rgba[..., 3]
    Hh, Ww = a.shape
    inside = (a > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
    gm = cv2.GaussianBlur(a, (0, 0), 2.0 * s + 0.6)
    gx, gy = np.gradient(gm, axis=1), np.gradient(gm, axis=0)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    xs = np.arange(Ww, dtype=np.float32)[None, :]
    lx = np.clip((x_warm - xs) / (0.12 * Wp), -1, 1)
    ly = -0.45
    ln = np.sqrt(lx * lx + ly * ly)
    facing = _ss(0.35, 0.9, np.clip((-gx * lx - gy * ly) / (gn * ln), 0, 1))
    near = 0.35 + 0.65 * np.exp(-np.abs(xs - x_warm) / (0.35 * Wp))
    rw = 1.1 * s + 0.3
    r = (_ss(rw * 2.0, rw * 0.8, dist) * 0.9 + np.exp(-dist / (rw * 4.0)) * 0.15) * (dist > 0) * facing * near * amt
    r = np.clip(r, 0, 0.8)
    out = rgba.copy()
    out[..., :3] = rgba[..., :3] * (1 - r[..., None]) + np.asarray(col, np.float32) * r[..., None]
    return out


def mist_layer(Wp, Hl, y_h, H, s, light_x, seed=61):
    """Straight RGBA (Hl rows): a wide, soft, low-contrast mist band over the valley floor
    (~0.03-0.11 H above the waterline), brightest on top where the comet light catches it and fading
    down into the forest; plus two torn wisps in front of the lower slopes of the near ranges."""
    from lib import clouds as K
    ys = np.arange(Hl, dtype=np.float32)[:, None]
    xs = np.arange(Wp, dtype=np.float32)[None, :]
    st = K.streaks(Wp, Hl, seed, xcells=5.0, ycells=36.0, octaves=4)
    st2 = K.streaks(Wp, Hl, seed + 1, xcells=9.0, ycells=70.0, octaves=3)
    yc = y_h - 0.07 * H
    # band: soft top, denser low, fading into the forest band below
    prof = np.exp(-((ys - yc) / (0.032 * H)) ** 2) * (ys < yc) + np.exp(-((ys - yc) / (0.05 * H)) ** 2) * (ys >= yc)
    lit0 = np.exp(-np.abs(xs - light_x) / (0.3 * Wp))
    a = prof * (0.36 + 0.26 * st) * (0.85 + 0.15 * np.cos(xs / Wp * 5.0 + 1.0)) * (0.9 + 0.45 * lit0)
    # torn wisps across the lower slopes (left massif, right ridge)
    for (xa, xb, yw, th, amt) in ((0.02, 0.34, y_h - 0.15 * H, 0.012 * H, 0.42),
                                  (0.7, 0.98, y_h - 0.125 * H, 0.01 * H, 0.36)):
        xm = C.smoothstep(xa * Wp, (xa + 0.06) * Wp, xs) * C.smoothstep(xb * Wp, (xb - 0.08) * Wp, xs)
        tilt = (xs - xa * Wp) * 0.02
        wy = np.exp(-((ys - yw - tilt) / th) ** 2)
        a = a + wy * xm * C.smoothstep(0.35, 0.8, st2) * amt
    a = np.clip(a, 0, 0.7).astype(np.float32)
    lit = np.exp(-np.abs(xs - light_x) / (0.35 * Wp))
    top = np.clip((yc - ys) / (0.05 * H), 0, 1)
    col = (np.array([0.24, 0.36, 0.64], np.float32) + (top * (0.6 + 0.4 * lit))[..., None] *
           np.array([0.1, 0.16, 0.22], np.float32) + lit[..., None] * np.array([0.03, 0.06, 0.08], np.float32))
    col = np.broadcast_to(col, (Hl, Wp, 3))
    return np.dstack([col, a]).astype(np.float32)
