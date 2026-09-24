"""More painted surface textures for s08_snow_station (round-2 props): weathered painted-steel relay
cabinet (grime runs, rust creeping up from the snow line, chipped paint on the edges, louvres, a real
Japanese warning label), painted-wood canopy pillars and bench slats (grain, dark board gaps, worn light
edges, dirt splash), and a wood grain / grime pass for the station name board frame and posts.
All functions return albedo textures (h, w, 3) that the caller multiplies with its own lighting."""
import numpy as np
import cv2

import s08_snow_station_tex as TX


def _n(seed, h, w, cy, cx):
    rng = np.random.default_rng(seed)
    g = rng.random((max(cy, 2), max(cx, 2))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _label(w, h, text, bg=(236, 196, 40), fg=(20, 20, 20)):
    from PIL import Image, ImageDraw
    import s08_snow_station_paint as PT
    S = 4
    im = Image.new('RGB', (max(w, 4) * S, max(h, 4) * S), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width - 1, im.height - 1], outline=fg, width=max(S, 2))
    d.text((im.width / 2, im.height / 2), text, font=PT._font(im.height * 0.62, True), fill=fg, anchor='mm')
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (max(w, 4), max(h, 4)), interpolation=cv2.INTER_AREA)


def steel_cabinet(tw, th, seed=0, base=(0.5, 0.56, 0.55), louvres=True, label=True):
    """Front of a trackside relay cabinet (two doors): painted steel with blotchy tone, vertical grime
    runs from the top, a rust band creeping up from the bottom, chipped light paint on the edges and
    door seams, louvre vents and a yellow '高圧注意' warning label. Returns (th, tw, 3) albedo."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, th, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, tw, dtype=np.float32)[None, :]
    base = np.asarray(base, np.float32)
    blot = _n(seed + 1, th, tw, 5, 4) - 0.5
    col = base * (1 + 0.16 * blot[..., None])
    # grime runs from the top edge and from under the louvres
    st = cv2.GaussianBlur(rng.random((th, tw)).astype(np.float32), (0, 0), sigmaX=max(tw * 0.008, 0.6), sigmaY=th * 0.12)
    st = (st - st.mean()) / (st.std() + 1e-6)
    runs = np.clip(st * 0.6 + 0.3, 0, 2) * (0.35 + 0.65 * np.exp(-y / 0.45))
    col = col * (1 - 0.22 * runs[..., None]) + np.array([0.16, 0.13, 0.1], np.float32) * 0.18 * runs[..., None]
    # rust creeping up from the snow line (ragged top edge), with rust runs from the hinge points
    edge = 0.84 + 0.05 * (_n(seed + 2, 1, tw, 2, 14)[0] - 0.5) * 2
    rust = np.clip((y - edge[None, :]) / 0.05, 0, 1) * (0.55 + 0.45 * _n(seed + 3, th, tw, 16, 12))
    for hx in (0.03, 0.49, 0.51, 0.97):
        for hy in (0.18, 0.72):
            rust += 0.7 * np.exp(-((x - hx) * tw / (0.012 * tw + 1.2)) ** 2) * np.clip((y - hy) / 0.01, 0, 1) * \
                np.exp(-(y - hy) / 0.16) * (0.6 + 0.4 * np.clip(st, -1, 2))
    rust = np.clip(rust, 0, 1)
    rc = np.array([0.46, 0.22, 0.1], np.float32) * (0.8 + 0.4 * _n(seed + 4, th, tw, 20, 20)[..., None])
    col = col * (1 - 0.8 * rust[..., None]) + rc * 0.8 * rust[..., None]
    # frame border and the door seam, with chipped light paint along the edges
    bw = 0.035
    border = (x < bw) | (x > 1 - bw) | (y < bw * tw / th) | (y > 1 - bw * tw / th)
    seam = np.abs(x - 0.5) < 0.006 + 0.5 / tw
    chips = (_n(seed + 5, th, tw, 40, 30) > 0.62)
    near_edge = (np.minimum(np.minimum(x, 1 - x), np.abs(x - 0.5)) * tw < 3.5) | (np.minimum(y, 1 - y) * th < 3.0)
    col = np.where(border[..., None], col * 0.82, col)
    col = np.where((chips & near_edge)[..., None], col * 0.4 + np.array([0.78, 0.8, 0.8], np.float32) * 0.6, col)
    col = np.where(seam[..., None], col * 0.25, col)
    # louvres on both doors: dark slot with a lit lower lip
    if louvres:
        for (x0, x1) in ((0.1, 0.4), (0.6, 0.9)):
            for k in range(5):
                yc = 0.58 + k * 0.045
                sl = (x > x0) & (x < x1) & (np.abs(y - yc) < 0.009)
                lip = (x > x0) & (x < x1) & (y > yc + 0.009) & (y < yc + 0.018)
                col = np.where(sl[..., None], col * 0.22, col)
                col = np.where(lip[..., None], col * 1.25, col)
    # handles
    for hx in (0.44, 0.56):
        hm = (np.abs(x - hx) < 0.012) & (np.abs(y - 0.45) < 0.05)
        col = np.where(hm[..., None], np.array([0.12, 0.12, 0.13], np.float32), col)
    if label and tw > 30:
        lw, lh = int(tw * 0.28), int(th * 0.08)
        if lh >= 5:
            lab = _label(lw, lh, '高圧注意')
            lab = lab * (1 - 0.18 * np.clip(_n(seed + 6, lh, lw, 3, 3) - 0.3, 0, 1)[..., None])
            ly, lx = int(th * 0.2), int(tw * 0.12)
            col[ly:ly + lh, lx:lx + lw] = lab * 0.85
    return np.clip(col, 0, 1.5).astype(np.float32)


def painted_wood(tw, th, Lm, Hm, seed=0, base=(0.36, 0.25, 0.17), per=None, along_x=True, grime_bottom=0.0,
                 worn_edges=True):
    """Painted / weathered wood panel of Lm x Hm metres rendered at (th, tw) px: grain along the boards,
    per-board tone, dark gaps between boards (per = board width, None = one board), lighter worn edges
    and chipped paint, dirt streaks, and a dirt splash `grime_bottom` (0..1) at the bottom."""
    base = np.asarray(base, np.float32)
    ppm = tw / max(Lm, 1e-3) if along_x else th / max(Hm, 1e-3)
    if along_x:
        U, V = np.meshgrid(np.linspace(0, Lm, tw, dtype=np.float32), np.linspace(Hm, 0, th, dtype=np.float32))
    else:
        V, U = np.meshgrid(np.linspace(0, Lm, tw, dtype=np.float32), np.linspace(Hm, 0, th, dtype=np.float32))
    per_ = per if per is not None else (Hm if along_x else Lm) * 1.0001
    m = TX.wood(U, V, per_, max(ppm, 60.0), seed=seed, joints=False)
    m = 1 + (m - 1) * 1.25                                    # a little stronger painted grain
    col = base * m[..., None]
    # board gaps
    if per is not None:
        fv = np.mod(V / per, 1.0)
        gap = np.clip(1 - np.minimum(fv, 1 - fv) * per * ppm / 1.2, 0, 1)
        col = col * (1 - 0.6 * gap[..., None])
    # worn light edges (top/bottom of the board or the pillar's sides) + chipped paint
    y = np.linspace(0, 1, th, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, tw, dtype=np.float32)[None, :]
    if worn_edges:
        e = np.minimum(y, 1 - y) * th if along_x else np.minimum(x, 1 - x) * tw
        wear = np.clip(1 - e / 2.5, 0, 1) * (_n(seed + 21, th, tw, 8, 30) > 0.45)
        col = col * (1 - 0.6 * wear[..., None]) + np.array([0.62, 0.52, 0.42], np.float32) * 0.6 * wear[..., None]
    # dirt / grime streaks running down and a splash at the bottom
    rng = np.random.default_rng(seed + 22)
    st = cv2.GaussianBlur(rng.random((th, tw)).astype(np.float32), (0, 0), sigmaX=max(tw * 0.01, 0.6),
                          sigmaY=max(th * 0.1, 1.0))
    st = (st - st.mean()) / (st.std() + 1e-6)
    col = col * (1 - 0.14 * np.clip(st, 0, 2)[..., None])
    if grime_bottom:
        sp = np.clip((y - 0.8) / 0.2, 0, 1) ** 1.5 * (0.6 + 0.4 * _n(seed + 23, th, tw, 6, 10))
        col = col * (1 - grime_bottom * 0.5 * sp[..., None]) + np.array([0.1, 0.08, 0.07], np.float32) * grime_bottom * 0.3 * sp[..., None]
    return np.clip(col, 0, 1.5).astype(np.float32)
