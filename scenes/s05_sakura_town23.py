"""s05_sakura round 23: far-bank town wear pass (reviewer: 'structure is there, but the shading is clean CG - uniform
grey balcony slabs and untextured plaster.  Add painted wear (rain streaks, stains), warm lit faces against cool
blue-violet shade on the side walls, AC units / laundry clutter, aerial fade on the furthest blocks').

Runs on the supersampled environment buffer after the shopfront pass (so it moves with the town's parallax):
  * warm / cool split per face orientation: the face family that is brighter on average (sun side) is warmed
    toward cream-apricot, the other family is pulled toward ONE cool blue-violet shade value;
  * rain streaks in world units: thin vertical grime runs hanging from the eaves / slab edges (hash columns,
    varied length), darker and a touch warmer-grey; low-frequency plaster stains; damp darker plinth band;
  * balcony slabs / railings (the pale long boxes) get the same wear plus a painted underside shadow band;
  * extra AC units + laundry on plain wall spans of the nearer blocks;
  * stronger blue aerial fade on the furthest blocks (starts nearer, reaches further).
"""
import numpy as np
import cv2


def _h(a, b, c=0.0):
    x = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return x - np.floor(x)


def _vn(rng, h, w, cell):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]


def wear(out, mat, bid, face, uo, Yo, so, layer, Zo, B, f, ss, seed=2323):
    rng = np.random.default_rng(seed)
    rgb = out[..., :3]
    Hs, Ws = mat.shape
    town = (mat == 5) & (layer == 0)
    if not town.any():
        return
    af = np.abs(face)
    fac = town & ((af == 1) | (af == 3))
    nb = len(B)
    bs = np.clip(bid, 0, nb - 1)
    lum = rgb.mean(-1)
    # ------------------------------------------------------------------ warm lit / cool shade face families
    fams = [c for c in np.unique(face[fac]) if c != 0]
    means = {int(c): float(lum[fac & (face == c)].mean()) for c in fams if (fac & (face == c)).sum() > 500}
    if means:
        lit_c = max(means, key=means.get)
        lit = fac & (face == lit_c)
        shd = fac & (face != lit_c)
        # sun side: warm cream-apricot, a touch brighter
        k = lit.astype(np.float32)[..., None]
        rgb[:] = rgb * (1 + k * np.array([0.06, 0.02, -0.06], np.float32))
        # shade side: one cool blue-violet value family (keeps the painted value, swaps the hue)
        k = shd.astype(np.float32)
        cool = np.array([0.66, 0.68, 0.9], np.float32)
        l_ = rgb.mean(-1, keepdims=True)
        tgt = cool / cool.mean() * l_ * 0.9
        rgb[:] = rgb + (tgt - rgb) * (0.4 * k)[..., None]
    # ------------------------------------------------------------------ world-space wear on every facade pixel
    ys, xs = np.nonzero(fac)
    if len(ys):
        b = bs[ys, xs]
        fc = af[ys, xs]
        along = np.where(fc == 1, so[ys, xs], uo[ys, xs])
        y = Yo[ys, xs]
        top = B[b, 3]
        base = B[b, 2]
        dtop = np.maximum(top - y, 0.0)
        hb = np.maximum(y - base, 0.0)
        # rain streaks: 0.22 m columns, each with its own start height (eave or a floor line) and run length
        cw = 0.22
        col = np.floor(along / cw)
        fr = along / cw - col
        h1 = _h(col, b.astype(np.float64), 1.0)
        h2 = _h(col, b.astype(np.float64), 2.0)
        h3 = _h(col, b.astype(np.float64), 3.0)
        start = np.floor(h2 * 3.0) * 2.8                      # below the eave or a floor slab (~2.8 m floors)
        run = 0.6 + 2.6 * h3
        dd = dtop - start
        inrun = (dd > 0) & (dd < run) & (h1 < 0.42)
        wid = 0.18 + 0.3 * h3
        prof = np.clip(1.0 - np.abs(fr - 0.5) / (0.5 * wid), 0, 1)
        fade = np.clip(1.0 - dd / np.maximum(run, 1e-3), 0, 1) ** 0.7
        streak = prof * fade * inrun
        # stains: low-frequency plaster blotches in world units (~1.4 m), damp plinth
        st = np.sin(along * 2.1 + b * 1.7) * np.sin(y * 1.6 + along * 0.7 + b * 0.3) + \
            0.5 * np.sin(along * 5.3 - y * 3.1 + b)
        stain = np.clip((st - 0.55) / 0.5, 0, 1)
        plinth = np.clip((0.7 - hb) / 0.3, 0, 1) * (hb >= 0)
        # eave soot band right under the roof line
        soot = np.clip((0.45 - dtop) / 0.35, 0, 1)
        c = rgb[ys, xs]
        grime = np.array([0.72, 0.7, 0.72], np.float32)
        dark = 0.3 * streak + 0.1 * stain + 0.14 * plinth + 0.16 * soot
        c = c * (1.0 - dark[:, None] * (1.0 - grime * 0.35))
        # lichen / rust tint in the streaks on warm walls
        c = c + (np.array([0.5, 0.42, 0.36], np.float32) * c.mean(-1, keepdims=True) * 1.2 - c) * \
            (0.12 * streak)[:, None]
        rgb[ys, xs] = c
    # ------------------------------------------------------------------ slab / railing boxes: underside shadow
    # (thin pale horizontal boxes -> the bottom 30% of their facade gets a cool shade band)
    slab = fac & ((B[bs, 3] - B[bs, 2]) < 2.0)
    if slab.any():
        ys, xs = np.nonzero(slab)
        b = bs[ys, xs]
        t = (Yo[ys, xs] - B[b, 2]) / np.maximum(B[b, 3] - B[b, 2], 1e-3)
        k = np.clip((0.35 - t) / 0.15, 0, 1)
        c = rgb[ys, xs]
        c = c + (np.array([0.42, 0.44, 0.6], np.float32) * c.mean(-1, keepdims=True) * 1.25 - c) * (0.5 * k)[:, None]
        # uneven painted value along the slab
        al = np.where(af[ys, xs] == 1, so[ys, xs], uo[ys, xs])
        c = c * (1.0 - 0.06 * (np.sin(al * 1.3 + b) * 0.5 + 0.5))[:, None]
        rgb[ys, xs] = c
    # ------------------------------------------------------------------ aerial fade on the furthest blocks
    far = town & (Zo > 120.0)
    if far.any():
        k = np.clip((Zo - 120.0) / 260.0, 0, 1) ** 0.8 * 0.4 * far
        hz = np.array([0.76, 0.84, 0.96], np.float32)
        rgb[:] = rgb + (hz - rgb) * k[..., None]
    out[..., :3] = np.clip(rgb, 0, 1.25)


def canal_wall(out, mat, Yo, so, Zo, y_water, y_far):
    """round 23: far embankment wall - warm / cool split instead of one lavender value: the upper band catches
    the low sun over the coping (warm ochre-grey, crisp wavy terminator = cast shadow of the bank railing / trees),
    the rest sits in cool blue shade; per-block tone variation (some stones warmer, some cooler), faint
    warm water-bounce glow just above the wet band."""
    m = mat == 3
    if not m.any():
        return
    rgb = out[..., :3]
    ys, xs = np.nonzero(m)
    Y = Yo[ys, xs]
    s = so[ys, xs]
    c = rgb[ys, xs]
    hgt = max(y_far - y_water, 0.5)
    t = (Y - y_water) / hgt                                  # 0 waterline .. 1 coping
    row = np.floor((Y - y_water) / 0.42)
    col = np.floor(s / 0.9 + 0.5 * (row % 2))
    hb = _h(row, col, 5.0)
    hb2 = _h(row, col, 6.0)
    lum = c.mean(-1, keepdims=True)
    # per-block tone: warm or cool stones
    warm_st = np.array([1.07, 1.0, 0.88], np.float32)
    cool_st = np.array([0.93, 0.97, 1.08], np.float32)
    k = (hb < 0.35)[:, None]
    c = np.where(k, c * warm_st, np.where((hb > 0.75)[:, None], c * cool_st, c))
    c = c * (0.94 + 0.1 * hb2)[:, None]
    # sunlit upper band with a wavy cast-shadow terminator
    term = 0.62 + 0.08 * np.sin(s * 0.35) + 0.05 * np.sin(s * 1.3 + 1.0) + 0.03 * np.sin(s * 3.7)
    lit = np.clip((t - term) / 0.03, 0, 1)
    sunc = np.array([0.98, 0.86, 0.7], np.float32)
    c = c + (sunc * np.maximum(lum, 0.5) * 1.25 - c) * (0.55 * lit)[:, None]
    # shade: one cool blue value family
    shd = 1.0 - lit
    coolc = np.array([0.5, 0.56, 0.74], np.float32)
    c = c + (coolc * lum / coolc.mean() - c) * (0.35 * shd)[:, None]
    # warm water-bounce glow above the wet band
    wb = np.exp(-((t - 0.28) / 0.12) ** 2) * shd
    c = c + np.array([0.08, 0.05, 0.0], np.float32) * wb[:, None]
    rgb[ys, xs] = c
    out[..., :3] = rgb
