"""s05_sakura round 7: painted line / accent pass over the far-bank town (screen space, on the supersampled env
render). Background painters draw architecture with crisp dark accents where surfaces meet (eave undersides,
window recesses, corners, object silhouettes) and thin light catch-lines along sun-facing top edges - that is
what separates a painted facade from a clean CG box. Also adds soft weathering (rain stains running down from
sills / parapets, darker plinths) and per-building value variation so no two walls read as the same flat plane.
"""
import numpy as np
import cv2


def ink(out, mat, bid, face, Zo, layer, ss, wscale, zmax=260.0, strength=1.0):
    rgb = out[..., :3]
    Hs, Ws = mat.shape
    bmask = (layer == 0) & (mat == 5)
    if not bmask.any():
        return
    key = bid.astype(np.int64) * 16 + face.astype(np.int64) + 8
    key = np.where(bmask, key, -1)
    Z = np.where(bmask, Zo, 1e5).astype(np.float32)
    # boundaries between different surfaces (object id or face)
    e = np.zeros((Hs, Ws), np.float32)
    dxk = key[:, 1:] != key[:, :-1]
    dyk = key[1:, :] != key[:-1, :]
    # the line belongs to the NEARER side of a depth step / both sides of a crease
    zr = Z[:, 1:] < Z[:, :-1]
    zd = Z[1:, :] < Z[:-1, :]
    e[:, 1:] = np.maximum(e[:, 1:], (dxk & zr).astype(np.float32))
    e[:, :-1] = np.maximum(e[:, :-1], (dxk & ~zr).astype(np.float32))
    e[1:, :] = np.maximum(e[1:, :], (dyk & zd).astype(np.float32))
    e[:-1, :] = np.maximum(e[:-1, :], (dyk & ~zd).astype(np.float32))
    # thicken to ~1 output px (ss px) and soften for AA
    k = max(1, int(round(0.7 * ss * wscale)))
    if k > 1:
        e = cv2.dilate(e, np.ones((k, k), np.uint8))
    e = cv2.GaussianBlur(e, (0, 0), 0.5 * ss * wscale)
    near = np.clip(1.0 - (Z - 25.0) / (zmax - 25.0), 0, 1)
    e = e * bmask * near * strength
    # dark accent: cool, darker multiply (never black)
    acc = np.array([0.62, 0.6, 0.72], np.float32)
    rgb *= (1 - 0.55 * e[..., None]) + acc * (0.55 * e[..., None])


def weather(out, mat, uo, Yo, so, face, layer, Zo, seed=3, amt=1.0):
    """soft vertical rain streaks + subtle per-surface value variation on the far-bank facades"""
    rgb = out[..., :3]
    Hs, Ws = mat.shape
    bmask = ((layer == 0) & (mat == 5)).astype(np.float32)
    if not bmask.any():
        return
    rng = np.random.default_rng(seed)
    # facade coordinates: horizontal = s (river face) or u (side faces), vertical = Y
    hcoord = np.where(np.abs(face) == 1, so, uo).astype(np.float32)
    # 1-D noise along the facade, stretched vertically -> streaks
    tab = rng.random(4096).astype(np.float32)
    tab = cv2.GaussianBlur(tab[None, :], (0, 0), 1.2)[0]
    idx = (np.mod(hcoord * 3.0, 4095.0)).astype(np.int32)
    st = tab[idx]
    st = np.clip((st - 0.55) / 0.3, 0, 1)
    # stains fade out downward from each floor line (2.8 m storeys)
    fy = np.mod(Yo + 1.3, 2.8) / 2.8
    st = st * (1 - fy) ** 1.5
    near = np.clip(1.0 - (Zo - 30.0) / 200.0, 0, 1)
    k = (0.1 * st * near * bmask * amt)[..., None]
    rgb *= 1 - k * np.array([1.0, 1.0, 0.7], np.float32)
    # painted wall mottle: low-frequency value / temperature variation (brush wash), facade-attached
    def nz(cell):
        gh, gw = int(Hs / cell) + 3, int(Ws / cell) + 3
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:Hs, :Ws]
    mt = 0.6 * nz(40.0) + 0.4 * nz(12.0)
    km = (0.035 * mt * near * bmask * amt)[..., None]
    rgb *= 1 + km * np.array([1.0, 0.9, 0.6], np.float32)


def dress2(B, seed=83):
    """round 7: extra lived-in clutter on the front row of far-bank buildings (same 16-column box layout as
    s05_sakura_env): upper-floor AC units with pipe ducts, amado shutter boxes beside the upper windows,
    tall water-heater units and meter boxes at street level, futons / laundry hung over balcony rails."""
    import s05_sakura_env as E
    rng = np.random.default_rng(seed)
    rows = []

    def add(u0, u1, Y0, Y1, s0, s1, typ, col, layer=0):
        rows.append([u0, u1, Y0, Y1, s0, s1, typ, col[0], col[1], col[2], rng.integers(0, 1000), layer,
                     0, 0, 0, 0])
    AC = (0.93, 0.93, 0.94)
    PIPE = (0.8, 0.8, 0.82)
    for i in range(B.shape[0]):
        if int(B[i, 6]) != 0 or int(B[i, 11]) != 0:
            continue
        u0, u1, Y0, Y1, s0, s1 = B[i, :6]
        if u1 < -45 or s0 > 130 or s1 < 0:
            continue
        w = s1 - s0
        hB = Y1 - Y0
        # upper-floor AC units + ducts
        for _ in range(int(rng.integers(1, 3))):
            fl = int(rng.integers(1, max(2, int(hB / 2.8))))
            y = Y0 + fl * 2.8 + 0.25
            if y + 0.7 > Y1 - 0.3:
                continue
            a = s0 + rng.uniform(0.4, max(0.5, w - 1.3))
            add(u1, u1 + 0.3, y, y + 0.6, a, a + 0.8, 8, AC)
            add(u1, u1 + 0.07, y + 0.25, y + 0.25 + rng.uniform(0.8, 1.8), a + 0.84, a + 0.92, 1, PIPE)
        # amado shutter boxes (house storeys)
        if hB < 8.5:
            for _ in range(int(rng.integers(1, 3))):
                a = s0 + rng.uniform(0.4, max(0.5, w - 0.8))
                y = Y0 + 2.8 + 0.5
                if y + 1.3 < Y1:
                    col = (0.78, 0.76, 0.72) if rng.random() < 0.5 else (0.52, 0.5, 0.5)
                    add(u1, u1 + 0.22, y, y + 1.3, a, a + 0.42, 9, col)
        # street level: tall water heater + meter box
        if rng.random() < 0.7:
            a = s0 + rng.uniform(0.3, max(0.4, w - 0.9))
            add(u1, u1 + 0.35, Y0, Y0 + 1.6, a, a + 0.6, 8, (0.9, 0.9, 0.88))
        if rng.random() < 0.6:
            a = s0 + rng.uniform(0.3, max(0.4, w - 0.6))
            add(u1, u1 + 0.1, Y0 + 1.3, Y0 + 1.75, a, a + 0.35, 8, (0.82, 0.84, 0.88))
        # futon / bedding airing over an upper rail
        if rng.random() < 0.35 and hB > 5.0:
            a = s0 + rng.uniform(0.3, max(0.4, w - 2.2))
            fl = int(rng.integers(1, max(2, int(hB / 2.8))))
            y = Y0 + fl * 2.8 + 0.2
            if y + 1.2 < Y1:
                col = [(0.97, 0.95, 0.9), (0.95, 0.8, 0.82), (0.72, 0.82, 0.95), (0.98, 0.9, 0.7)][int(rng.integers(4))]
                add(u1 + 0.3, u1 + 0.4, y, y + 1.1, a, a + rng.uniform(1.4, 2.0), 14, col)
    if not rows:
        return B
    return np.concatenate([B, np.array(rows, np.float64)], 0)
