"""Tight lamp cones for s08_snow_station: single-scattering airlight and per-flake lighting with a
shaped spotlight cone (inner/outer cosine, smooth falloff, small omni residual that only lights the air
right around the lamp head). Replaces the broad, soft cone of the first pass so each lamp throws a
clean amber beam into the falling snow instead of a wide smear."""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline='always')
def _cone(cd, cin, cout, omni, pw):
    if cout >= cin:
        return 1.0
    c = (cd - cout) / (cin - cout)
    if c < 0.0:
        c = 0.0
    elif c > 1.0:
        c = 1.0
    c = c * c * (3.0 - 2.0 * c)
    return omni + (1.0 - omni) * c ** pw


@njit(cache=True, fastmath=True, parallel=True)
def inscatter(dx, dy, depth, lp, lcol, cin, cout, omni, pw, ext, zmax, nsamp):
    """Airlight along each view ray (camera at the origin). dx, dy: ray slopes X/Z, Y/Z; depth: scene Z.
    lp (L,3) lamp positions (camera space), lcol (L,3) colour*intensity; cone per lamp: cin/cout = cos of
    inner/outer half angle about straight down (cout >= cin -> omni), omni residual, pw sharpness;
    ext (L,) extinction length (m). Returns (H, W, 3)."""
    H, W = dx.shape
    out = np.zeros((H, W, 3), np.float32)
    nl = lp.shape[0]
    for i in prange(H):
        for j in range(W):
            rx, ry = dx[i, j], dy[i, j]
            rn = math.sqrt(rx * rx + ry * ry + 1.0)
            ux, uy, uz = rx / rn, ry / rn, 1.0 / rn
            T = min(depth[i, j], zmax) * rn
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            for k in range(nl):
                e = ext[k]
                px, py, pz = lp[k, 0], lp[k, 1], lp[k, 2]
                s0 = px * ux + py * uy + pz * uz
                qx, qy, qz = px - s0 * ux, py - s0 * uy, pz - s0 * uz
                h2 = qx * qx + qy * qy + qz * qz
                if h2 > (8.0 * e) ** 2:
                    continue
                sa = max(s0 - 8.0 * e, 0.0)
                sb = min(s0 + 8.0 * e, T)
                if sb <= sa:
                    continue
                he = math.sqrt(h2) + 0.03
                ta = math.atan((sa - s0) / he)
                tb = math.atan((sb - s0) / he)
                dth = (tb - ta) / nsamp
                acc = 0.0
                for m in range(nsamp):
                    th = ta + (m + 0.5) * dth
                    ss = s0 + he * math.tan(th)
                    ds = he / (math.cos(th) ** 2) * dth
                    wx = ss * ux - px
                    wy = ss * uy - py
                    wz = ss * uz - pz
                    d2 = wx * wx + wy * wy + wz * wz
                    d = math.sqrt(d2) + 1e-4
                    v = math.exp(-d / e) / (d2 + 0.08)
                    v *= _cone(-wy / d, cin[k], cout[k], omni[k], pw[k])
                    acc += v * ds
                a0 += acc * lcol[k, 0]
                a1 += acc * lcol[k, 1]
                a2 += acc * lcol[k, 2]
            out[i, j, 0] = a0
            out[i, j, 1] = a1
            out[i, j, 2] = a2
    return out


@njit(cache=True, fastmath=True, parallel=True)
def flake_light(P, cam, lp, lcol, lI, cin, cout, omni, pw, g):
    """Per-flake lamp irradiance E (N,3) (inside the shaped cones) and forward-scattered glow FS (N,3)."""
    n = P.shape[0]
    nl = lp.shape[0]
    E = np.zeros((n, 3), np.float32)
    FS = np.zeros((n, 3), np.float32)
    for i in prange(n):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        vx, vy, vz = px - cam[0], py - cam[1], pz - cam[2]
        vn = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
        vx /= vn
        vy /= vn
        vz /= vn
        for k in range(nl):
            dx, dy, dz = lp[k, 0] - px, lp[k, 1] - py, lp[k, 2] - pz
            d2 = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(d2) + 1e-4
            cf = _cone(dy / dist, cin[k], cout[k], omni[k], pw[k])
            e = lI[k] / (d2 + 0.35) * cf
            cth = (vx * dx + vy * dy + vz * dz) / dist
            hg = (1 - g * g) / (1 + g * g - 2 * g * cth) ** 1.5 / 12.0
            f = lI[k] * hg / (d2 + 0.5) * cf
            for c3 in range(3):
                E[i, c3] += e * lcol[k, c3]
                FS[i, c3] += f * lcol[k, c3]
    return E, FS
