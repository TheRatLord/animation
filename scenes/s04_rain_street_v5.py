"""s04_rain_street round-5 helpers: crisp wet-road reflections (world-anchored wobble, specular hot spots,
ground light pools), foreground neon rain, drifting vanishing-point mist, rain-haze halation, splash
crowns on sign tops / at the walker's feet."""
import math

import numpy as np
import cv2

from lib import core as C
import s04_rain_street_art as A
import s04_rain_street_r as R

hexc = A.hexc

VEND_Z = (4.5, 5.5, 6.5)
VEND_XF = -3.5 + 0.75
VEND_TINT = ('#ff6a7a', '#f4f8ff', '#5a9aff')


# ----------------------------------------------------------------------------- ground light pools
def ground_pools(X, Z, below):
    """Emissive light pooled on the wet pavement in front of the vending machines (display glow + button
    panel colours). X, Z: world coords per (supersampled) ground pixel."""
    out = np.zeros(X.shape + (3,), np.float32)
    front = X > VEND_XF - 0.02
    dx = np.clip(X - VEND_XF, 0, None)
    for z0, tint in zip(VEND_Z, VEND_TINT):
        z1 = z0 + 0.98
        wz = C.smoothstep(z0 - 0.35, z0 + 0.1, Z) * C.smoothstep(z1 + 0.35, z1 - 0.1, Z)
        fall = np.exp(-dx / 0.55) * 0.6 + np.exp(-dx / 0.14) * 0.9
        col = hexc('#dff0ff') * 0.75 + hexc(tint) * 0.25
        out += (fall * wz * front)[..., None] * col * 0.34
        # coin / button panel: small warm-green LCD + cyan buttons, a tight bright bleed at the kerb
        zp = z0 + 0.98 * 0.2
        panel = np.exp(-((Z - zp) / 0.12) ** 2) * np.exp(-dx / 0.1) * front
        out += panel[..., None] * hexc('#60ffb0') * 0.35
    return (out * below[..., None]).astype(np.float32)


# ----------------------------------------------------------------------------- wobble
def wobble(sc, t, dZ):
    """Horizontal displacement (half-res grid, full-res px) of the mirrored image: gentle swell rows that are
    anchored to the road (world Z) and roll toward the camera. Smooth over tens of px vertically, so the
    reflected lettering stays legible."""
    if not hasattr(sc, '_wv'):
        cam = sc.cam
        W2, H2 = sc.W // 2, sc.H // 2
        yy = (np.arange(H2, dtype=np.float32) + 0.5) * 2
        v = yy - cam.cy
        Zr = np.where(v > 2, cam.f * cam.h / np.maximum(v, 2), 300.0).astype(np.float32)
        g = np.clip(v / (sc.H - cam.cy), 0, 1)
        amp = ((0.35 + 3.4 * g ** 1.2) * sc.W / 1920.0 * (v > 0)).astype(np.float32)
        ph1 = A.noise(max(W2 // 8, 8), max(H2 // 8, 8), 5, 5151, 3)
        ph1 = cv2.resize(ph1, (W2, H2), interpolation=cv2.INTER_CUBIC)
        ph2 = A.noise(max(W2 // 8, 8), max(H2 // 8, 8), 7, 5152, 3)
        ph2 = cv2.resize(ph2, (W2, H2), interpolation=cv2.INTER_CUBIC)
        ph3 = A.noise(max(W2 // 4, 8), max(H2 // 16, 8), 11, 5153, 3)
        ph3 = cv2.resize(ph3, (W2, H2), interpolation=cv2.INTER_CUBIC)
        sc._wv = (Zr[:, None], amp[:, None], (ph1 * 7.0).astype(np.float32), (ph2 * 9.0).astype(np.float32),
                  (ph3 * 12.0).astype(np.float32))
    Zr, amp, p1, p2, p3 = sc._wv
    Zw = Zr + np.float32(dZ)
    w = (np.sin(Zw * np.float32(2 * np.pi / 0.85) + np.float32(2.4 * t) + p1) * np.float32(0.65) +
         np.sin(Zw * np.float32(2 * np.pi / 0.33) + np.float32(3.7 * t) + p2) * np.float32(0.35) +
         # fine horizontal ripple shimmer (short wavelength rows, rolling toward the camera)
         np.sin(Zw * np.float32(2 * np.pi / 0.09) + np.float32(6.0 * t) + p3) * np.float32(0.55))
    return (w * amp).astype(np.float32)


# ----------------------------------------------------------------------------- specular hot spots
def _hot_list(sc):
    L = []
    for i, (xw, z) in enumerate([(3.85, 15.0), (3.85, 17.8), (-3.35, 10.2), (-3.35, 12.4), (-3.4, 30.5),
                                 (-3.4, 32.2)]):
        L.append((xw - np.sign(xw) * 0.35, 2.0, z, hexc('#ff6a3a') * 1.6, 0.34))
    for (X, Z, hh, lamp) in sc.poles:
        if lamp and Z < 70:
            L.append((X - np.sign(X) * 1.0, 5.6, Z, hexc('#e4f2ff') * 2.4, 0.5))
    for (X, Z) in ((3.05, 9.2), (-2.75, 16.5)):
        L.append((X, 0.65, Z, hexc('#fff0e0') * 1.0, 0.45))
    L.append((1.5, 5.0, 27.6, hexc('#ff3a20') * 1.8, 0.4))
    for z0 in VEND_Z:
        L.append((VEND_XF, 1.2, z0 + 0.5, hexc('#e8f4ff') * 0.9, 0.9))
    return L


def _glint_sprite(r):
    r = max(r, 0.6)
    hw = int(math.ceil(r * 3)) + 1
    hh = int(math.ceil(r * 9)) + 1
    y, x = np.mgrid[-hh:hh + 1, -hw:hw + 1].astype(np.float32)
    core = np.exp(-(x / r) ** 2 - (y / (r * 2.2)) ** 2)
    tail = np.exp(-(x / (r * 0.6)) ** 2 - (y / (r * 7.0)) ** 2) * 0.35
    return core + tail


def hotspots(sc, img, gex, dX, dY, dZ):
    if not hasattr(sc, '_hot'):
        sc._hot = _hot_list(sc)
    cam = sc.cam
    H, W = img.shape[:2]
    K = R.MIRROR_K
    for (X, Y, Z, col, size) in sc._hot:
        zc = Z - dZ
        if zc < 0.8:
            continue
        x = cam.cx + cam.f * (X - dX) / zc
        y = cam.cy + cam.f * (cam.h + dY + K * Y) / zc
        if not (-30 < x < W + 30 and cam.cy < y < H + 30):
            continue
        r = cam.f * size * 0.1 / zc
        spr = _glint_sprite(r)
        sh, sw = spr.shape
        x0, y0 = int(round(x)) - sw // 2, int(round(y)) - sh // 2
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + sw, W), min(y0 + sh, H)
        if xb <= xa or yb <= ya:
            continue
        xi, yi = int(min(max(x, 0), W - 1)), int(min(max(y, 0), H - 1))
        e = gex[yi, xi]
        wet = max(e[2] * e[0], e[5] * (1 - e[4]) * (1 - e[0]), 0.15)
        img[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0, None] * (col * wet * 0.9)


# ----------------------------------------------------------------------------- drugstore front
def drugstore(As, Es, rng, ww, hh, fas, P):
    """Bright Tokyo drugstore front: white lightbox fascia with red copy, cool-white interior with fluorescent
    tube rows, dense shelves of small colourful boxes with yellow price POP, hanging promo banners, posters
    and a counter with a clerk silhouette. Glass carries street reflections and condensation."""
    import s04_rain_street_text as JT
    from s04_rain_street_art2 import fill, droplets
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None]
    As[:] = hexc('#2a2c34')
    # fascia: white lightbox, red band, real copy
    A.rect(As, 0, 0, ww, fas, hexc('#f4f6f8'))
    A.rect(Es, 0, 0, ww, fas, hexc('#fff4f4') * 0.55)
    A.rect(As, 0, fas * 0.78, ww, fas, hexc('#d81830'))
    A.rect(Es, 0, fas * 0.78, ww, fas, hexc('#ff2840') * 0.9)
    if fas > 6:
        seg = int(min(ww * 0.45, fas * 4.6))
        g = JT.row(rng, seg, int(fas * 0.6), 7, 0.06, text='ドラッグストア')
        gh, gw = g.shape
        for x0 in np.arange(ww - gw - P(0.25), 0, -(gw + P(0.5))):
            sl = (slice(int(fas * 0.08), int(fas * 0.08) + gh), slice(int(x0), int(x0) + gw))
            As[sl] = As[sl] * (1 - g[..., None]) + hexc('#d81830') * g[..., None]
            Es[sl] = Es[sl] * (1 - g[..., None]) + hexc('#ff3048') * 0.55 * g[..., None]
    gy0, gy1 = fas + P(0.06), hh - P(0.3)
    gx0, gx1 = P(0.15), ww - P(0.15)
    H_, W_ = int(gy1 - gy0), int(gx1 - gx0)
    if H_ < 4 or W_ < 4:
        return
    ys = np.arange(H_, dtype=np.float32)[:, None]
    xs = np.arange(W_, dtype=np.float32)[None, :]
    inter = np.ones((H_, W_, 3), np.float32) * hexc('#eef4ff') * (1.6 - 0.5 * ys / H_)[..., None]
    # fluorescent tube rows on the ceiling (perspective-less: bright bars near the top)
    for k in range(3):
        yk = P(0.08 + 0.13 * k)
        A.rect(inter, 0, yk, W_, yk + max(P(0.035 - 0.008 * k), 1), hexc('#ffffff') * (3.4 - 0.6 * k))
    # shelves of products
    sh_top = P(0.5)
    rowh = P(0.3)
    cols = ['#e83040', '#2a70e0', '#f2c230', '#28a860', '#f07a28', '#b04ad8', '#f4f4f4', '#20b8c8', '#ff7ab0',
            '#1a1a28', '#c8e84a', '#8a5a30']
    r = 0
    while sh_top + (r + 1) * rowh < H_ - P(0.2):
        base = sh_top + (r + 1) * rowh
        x = rng.uniform(0, P(0.05))
        while x < W_:
            bw = P(rng.uniform(0.05, 0.14))
            bh = rowh * rng.uniform(0.45, 0.85)
            c = hexc(rng.choice(cols)) * rng.uniform(0.85, 1.15)
            run = int(rng.integers(1, 6))            # facings of the same product
            for q in range(run):
                x1 = x + bw - max(P(0.006), 0.5)
                A.rect(inter, x, base - bh, x1, base, c * 1.25)
                A.rect(inter, x, base - bh, x1, base - bh + max(bh * 0.12, 1), c * 1.7 + 0.3)
                if bw > 4:
                    A.rect(inter, x + bw * 0.2, base - bh * 0.6, x1 - bw * 0.2, base - bh * 0.4, hexc('#ffffff') * 1.4)
                x += bw
            x += P(rng.uniform(0.0, 0.02))
        A.rect(inter, 0, base, W_, base + max(P(0.03), 1), hexc('#dfe6f0') * 1.3)
        # yellow price POP strip with red digits
        for xk in np.arange(rng.uniform(0, P(0.4)), W_, P(rng.uniform(0.35, 0.6))):
            A.rect(inter, xk, base + P(0.03), xk + P(0.12), base + P(0.075), hexc('#ffe020') * 1.6)
            A.rect(inter, xk + P(0.02), base + P(0.04), xk + P(0.09), base + P(0.065), hexc('#e01020') * 1.2)
        r += 1
    # hanging promo banners (red/yellow) from the ceiling
    for k in range(int(W_ / P(1.7)) + 1):
        bx = rng.uniform(0, W_ - P(0.5))
        c = hexc(rng.choice(['#ffd820', '#ff3040', '#ff80b0']))
        A.rect(inter, bx, P(0.12), bx + P(0.5), P(0.42), c * 1.5)
        if P(0.5) > 12:
            g = JT.row(rng, int(P(0.46)), int(P(0.2)), 3, 0.05, pool='shop', text=rng.choice(['特売', '半額', '新商品']))
            gh, gw = g.shape
            sl = (slice(int(P(0.17)), int(P(0.17)) + gh), slice(int(bx + P(0.02)), int(bx + P(0.02)) + gw))
            if inter[sl].shape[:2] == g.shape:
                inter[sl] = inter[sl] * (1 - g[..., None]) + hexc('#202020') * g[..., None]
    # counter + clerk at the far end
    cx0 = W_ - P(1.6)
    if cx0 > 0:
        A.rect(inter, cx0, H_ - P(1.05), W_, H_, hexc('#c8ccd4'))
        A.rect(inter, cx0, H_ - P(1.05), W_, H_ - P(1.0), hexc('#ffffff') * 2)
        m = np.zeros((H_, W_), np.float32)
        px, py = cx0 + P(0.8), H_ - P(1.05)
        cv2.ellipse(m, (int(px), int(py - P(0.62))), (max(1, int(P(0.1))), max(1, int(P(0.12)))), 0, 0, 360, 1.0, -1,
                    cv2.LINE_AA)
        fill(m, [(px - P(0.24), py), (px - P(0.2), py - P(0.42)), (px - P(0.08), py - P(0.5)),
                 (px + P(0.08), py - P(0.5)), (px + P(0.2), py - P(0.42)), (px + P(0.24), py)])
        inter = inter * (1 - m[..., None]) + hexc('#2a4a7a') * 0.6 * m[..., None]
    # mullions + posters on the glass
    nm = max(1, int(W_ / P(1.8)))
    for i in range(nm + 1):
        xm = W_ * i / nm
        A.rect(inter, xm - P(0.035), 0, xm + P(0.035), H_, hexc('#3a3c46'))
    for k in range(int(rng.integers(1, 3))):
        px = rng.uniform(0.05, 0.8) * W_
        py = rng.uniform(0.25, 0.45) * H_
        c = hexc(rng.choice(['#ff3a5a', '#2a80ff', '#ffd030']))
        A.rect(inter, px, py, px + P(0.45), py + P(0.62), c * 1.1)
        A.rect(inter, px + P(0.04), py + P(0.3), px + P(0.41), py + P(0.58), hexc('#ffffff') * 1.1)
    # condensation low on the glass, a few drip channels
    cond = np.clip((ys - (H_ - P(0.8))) / P(0.8), 0, 1) ** 1.2
    inter = inter * (1 - 0.35 * cond[..., None]) + hexc('#e8f0ff') * (0.5 * cond)[..., None]
    sub = (slice(int(gy0), int(gy0) + H_), slice(int(gx0), int(gx0) + W_))
    Es[sub] = Es[sub] + inter * 0.3
    # lightbox fascia: uneven tube hot-spots, rain run-off from the top edge and grime (painted, not a clean
    # CG white plane)
    fy = np.arange(int(fas) + 1, dtype=np.float32)[:, None] / max(fas, 1)
    fx = np.arange(ww, dtype=np.float32)[None, :]
    nt = max(2, int(ww / max(P(1.2), 1)))
    tubes = 0.72 + 0.28 * (0.5 + 0.5 * np.cos(fx / max(ww / nt, 1) * 2 * np.pi)) * np.exp(-((fy - 0.45) / 0.5) ** 2)
    cn = A.noise(max(ww // 3, 4), 6, max(4, int(ww / max(P(0.35), 1))), int(rng.integers(1e9)), 3)
    cn = cv2.resize(cn, (ww, int(fas) + 1), interpolation=cv2.INTER_CUBIC)
    runs = np.clip((cn - 0.5) * 3.5, 0, 1) * np.clip(1.1 - fy * 0.9, 0, 1)
    bl = A.noise(ww, int(fas) + 1, max(4, ww // 40), int(rng.integers(1e9)), 3)
    dirt = (0.82 + 0.3 * bl) * (1 - 0.45 * runs)
    k = (tubes * dirt)[..., None].astype(np.float32)
    Es[:int(fas) + 1] *= k * 0.8
    As[:int(fas) + 1] *= (0.75 + 0.25 * dirt)[..., None]
    As[sub] = inter * 0.05
    A.rect(As, 0, hh - P(0.3), ww, hh, hexc('#1e2026'))


# ----------------------------------------------------------------------------- shutter graffiti
def graffiti(As, Es, rng, x0, y0, x1, y1, P):
    """Spray-painted bubble piece (abstract letterforms, no fake text) + tag scribbles, drips and stickers on a
    lowered shutter. Slightly faded by rain; painted into the ribbed albedo."""
    from s04_rain_street_art2 import fill
    h, w = As.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    if bw < 20 or bh < 20:
        return
    body = np.zeros((h, w), np.float32)
    n = int(rng.integers(3, 6))
    cx0 = x0 + bw * rng.uniform(0.08, 0.2)
    span = bw * rng.uniform(0.55, 0.75)
    cy = y0 + bh * rng.uniform(0.35, 0.5)
    for i in range(n):
        cx = cx0 + span * (i + 0.5) / n
        rx = span / n * rng.uniform(0.45, 0.62)
        ry = bh * rng.uniform(0.18, 0.28)
        ang = rng.uniform(-15, 15)
        cv2.ellipse(body, (int(cx * 4), int((cy + rng.uniform(-0.05, 0.05) * bh) * 4)), (int(rx * 4), int(ry * 4)),
                    ang, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
        # counter-shape (letter hole)
        if rng.random() < 0.6:
            cv2.ellipse(body, (int(cx * 4), int(cy * 4)), (max(1, int(rx * 0.25 * 4)), max(1, int(ry * 0.35 * 4))),
                        ang, 0, 360, 0.0, -1, cv2.LINE_AA, 2)
    k = max(1, int(P(0.06)))
    outl = cv2.dilate(body, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) - body
    c1 = hexc(rng.choice(['#ff4aa0', '#3ad0ff', '#ffd23a', '#7aff6a', '#b06aff']))
    c2 = hexc(rng.choice(['#3a50ff', '#ff6a2a', '#ff3a6a', '#20c0a0']))
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    grad = c1 * (1 - yy) + c2 * yy
    fade = 0.8
    As[:] = As * (1 - fade * body[..., None]) + grad * 0.85 * fade * body[..., None]
    As[:] = As * (1 - fade * outl[..., None]) + hexc('#101018') * fade * outl[..., None]
    # white shine strokes on the bubbles
    sh = np.zeros((h, w), np.float32)
    for i in range(n):
        cx = cx0 + span * (i + 0.3) / n
        cv2.ellipse(sh, (int(cx * 4), int((cy - bh * 0.1) * 4)), (max(1, int(span / n * 0.15 * 4)), max(1, int(P(0.03) * 4))),
                    -20, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
    sh *= body
    As[:] = As * (1 - sh[..., None]) + hexc('#ffffff') * 0.9 * sh[..., None]
    Es[:] = Es + grad * (body * 0.05)[..., None]
    # drips below the piece
    for _ in range(int(rng.integers(3, 8))):
        dx = cx0 + rng.uniform(0, span)
        dy = cy + bh * 0.2
        L = bh * rng.uniform(0.08, 0.3)
        cv2.line(As, (int(dx * 4), int(dy * 4)), (int(dx * 4), int((dy + L) * 4)), tuple(float(c) for c in c2 * 0.7),
                 max(1, int(P(0.02))), cv2.LINE_AA, 2)
    # tag scribble (black marker loops)
    tx, ty = x0 + bw * rng.uniform(0.6, 0.8), y0 + bh * rng.uniform(0.7, 0.85)
    pts = []
    for q in range(14):
        pts.append((tx + q * P(0.05) + np.sin(q * 1.7) * P(0.04), ty + np.cos(q * 2.3) * P(0.08)))
    cv2.polylines(As, [np.int32(np.array(pts) * 4)], False, (0.05, 0.05, 0.07), max(1, int(P(0.015))), cv2.LINE_AA, 2)
    # stickers
    for _ in range(int(rng.integers(2, 5))):
        sx, sy = x0 + rng.uniform(0, bw * 0.9), y0 + rng.uniform(0, bh * 0.5)
        c = hexc(rng.choice(['#f0f0f0', '#ffe040', '#ff5050', '#40a0ff']))
        A.rect(As, sx, sy, sx + P(0.12), sy + P(0.08), c)


# ----------------------------------------------------------------------------- wire drops
def wire_drops(cv, P, dia, tint, rng):
    """Small water beads hanging under a wire (every ~0.3-0.7 m): a tiny dark bead with a bright neon-lit pip,
    drawn in one local mask."""
    cam = cv.cam
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < 0.5:
        return
    s = rng.uniform(0.1, 0.5)
    ds = []
    while s < L[-1]:
        ds.append(s)
        s += rng.uniform(0.7, 1.6)
    ds = np.array(ds)
    Q = np.stack([np.interp(ds, L, P[:, k]) for k in range(3)], 1)
    Q[:, 1] -= dia * 0.5 + 0.012
    Q = Q[Q[:, 2] > 0.8]
    if len(Q) == 0:
        return
    xs, ys = cam.proj(Q[:, 0], Q[:, 1], Q[:, 2])
    rr = 0.012 * cam.f * cam.ss / Q[:, 2]
    keep = rr > 0.35
    if not keep.any():
        return
    xs, ys, rr, zz = xs[keep], ys[keep], rr[keep], Q[keep, 2]
    x0, y0 = int(max(xs.min() - 6, 0)), int(max(ys.min() - 6, 0))
    x1, y1 = int(min(xs.max() + 7, cv.W)), int(min(ys.max() + 7, cv.H))
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    pip = np.zeros_like(m)
    for x, y, r in zip(xs, ys, rr):
        r2 = max(r, 0.6)
        cv2.ellipse(m, (int((x - x0) * 4), int((y - y0) * 4)), (int(r2 * 4), int(r2 * 1.25 * 4)), 0, 0, 360,
                    min(1.0, r / 0.6), -1, cv2.LINE_AA, 2)
        cv2.circle(pip, (int((x - x0 - r2 * 0.25) * 4), int((y - y0 - r2 * 0.2) * 4)), max(1, int(r2 * 0.5 * 4)),
                   min(1.0, r / 0.6), -1, cv2.LINE_AA, 2)
    Z = np.full(m.shape, float(np.median(zz)), np.float32)
    col = np.broadcast_to(hexc('#20243a'), m.shape + (3,))
    E = (pip[..., None] * (tint * 0.6 + 0.4) + m[..., None] * tint * 0.12).astype(np.float32)
    E = E / np.maximum(m[..., None], 1e-3)
    cv.put(x0, y0, col, np.clip(m, 0, 1), Z, np.float32([0, -0.3, -1]), np.minimum(E, 6.0))


# ----------------------------------------------------------------------------- vending machine wear & wet
def vend_extras(T, rng, brand):
    """Weathering on a vending machine front texture: cold/hot label strips, stickers, scuffs and rust at the
    kick plate, glowing coin-return / button bezel, heavy rain beading with long drip runs on the glass and a
    wet specular edge line down both front corners."""
    from s04_rain_street_art2 import droplets
    import s04_rain_street_text as JT
    Al, E = T['alb'], T['emi']
    h, w = T['a'].shape
    s = h / 1000.0
    x0, x1 = 0.065 * w, 0.935 * w
    y0, y1 = 0.075 * h, 0.6 * h
    rh = (y1 - y0) / 3
    # つめた〜い (blue) / あったか〜い (red) strips under the price labels
    for r in range(3):
        sy = y0 + (r + 1) * rh - rh * 0.28
        hot = (r == 2) and rng.random() < 0.6
        c = hexc('#e8202a') if hot else hexc('#1a6aff')
        A.rect(Al, x0, sy + rh * 0.105, x1, sy + rh * 0.13, c)
        A.rect(E, x0, sy + rh * 0.105, x1, sy + rh * 0.13, c * 0.9)
        if w > 150:
            g = JT.row(rng, int((x1 - x0) * 0.3), max(3, int(rh * 0.024)), 6, 0.02,
                       text='あったか〜い' if hot else 'つめた〜い')
            gh, gw = g.shape
            for xk in (x0 + (x1 - x0) * 0.05, x0 + (x1 - x0) * 0.6):
                sl = (slice(int(sy + rh * 0.106), int(sy + rh * 0.106) + gh), slice(int(xk), int(xk) + gw))
                if Al[sl].shape[:2] == g.shape:
                    Al[sl] = Al[sl] * (1 - g[..., None]) + g[..., None]
                    E[sl] = E[sl] + g[..., None] * 1.2
    # stickers on the lower cabinet (e-money logo, recycle notice, small warning)
    for k in range(int(rng.integers(2, 4))):
        sx = rng.uniform(0.08, 0.5) * w
        sy = rng.uniform(0.84, 0.86) * h if k == 0 else rng.uniform(0.62, 0.8) * h
        sw_, sh_ = rng.uniform(0.07, 0.14) * w, rng.uniform(0.02, 0.04) * h
        c = hexc(rng.choice(['#f4f4f4', '#ffd820', '#20c0ff', '#ff6aa0', '#40d070']))
        A.rect(Al, sx, sy, sx + sw_, sy + sh_, c)
        A.rect(Al, sx + sw_ * 0.1, sy + sh_ * 0.35, sx + sw_ * 0.7, sy + sh_ * 0.6, hexc('#202028'))
        A.rect(E, sx, sy, sx + sw_, sy + sh_, c * 0.12)
        # half-peeled corner
        A.rect(Al, sx + sw_ * 0.85, sy, sx + sw_, sy + sh_ * 0.3, hexc('#d8d8d8'))
    # scuffs, scratches and rust along the kick plate and lower corners
    n = A.noise(w, h, max(6, w // 10), int(rng.integers(1e9)), 4)
    low = np.clip((np.linspace(0, 1, h, dtype=np.float32)[:, None] - 0.8) / 0.2, 0, 1)
    scuff = np.clip((n - 0.55) * 4, 0, 1) * low
    Al *= (1 - 0.45 * scuff)[..., None]
    Al += hexc('#6a3a20') * (np.clip((n - 0.7) * 5, 0, 1) * low * 0.5)[..., None]
    for _ in range(int(14 * max(s, 0.3))):
        sx, sy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.62, 0.98) * h
        L = rng.uniform(0.02, 0.08) * w
        a = rng.uniform(-0.4, 0.4)
        cv2.line(Al, (int(sx * 4), int(sy * 4)), (int((sx + L * math.cos(a)) * 4), int((sy + L * math.sin(a)) * 4)),
                 (0.75, 0.76, 0.8), 1, cv2.LINE_AA, 2)
    # glowing coin-return / button bezel + LCD halo
    A.rect(E, 0.735 * w, 0.695 * h, 0.875 * w, 0.72 * h, hexc('#ffb040') * 0.9)
    A.rect(E, 0.69 * w, 0.632 * h, 0.91 * w, 0.683 * h, hexc('#50ff9a') * 0.35)
    # heavy rain beading on the display glass, with long drip runs
    droplets(Al, E, rng, x0, y0, x1, y1, 120 * (w * h) / 2.5e6 + 30, 1.6 * s, 4.2 * s, light=hexc('#ffffff'),
             trails=0.5, gain=0.9)
    droplets(Al, E, rng, 0, y1, w, 0.96 * h, 70 * (w * h) / 2.5e6 + 16, 1.4 * s, 3.4 * s, light=hexc('#f0f4ff'),
             trails=0.5, gain=0.7)
    tr = np.zeros((h, w), np.float32)
    for _ in range(int(26 * max(w / 400, 0.3))):
        xk = rng.uniform(0.02, 0.98) * w
        ya = rng.uniform(0.0, 0.5) * h
        yb = ya + rng.uniform(0.15, 0.5) * h
        pts = [(xk + math.sin(q * 0.9 + xk) * 1.2, ya + (yb - ya) * q / 12) for q in range(13)]
        cv2.polylines(tr, [np.int32(np.array(pts) * 4)], False, 1.0, max(1, int(1.6 * s)), cv2.LINE_AA, 2)
        cv2.circle(tr, (int(pts[-1][0] * 4), int(yb * 4)), max(1, int(3.0 * s * 4)), 1.0, -1, cv2.LINE_AA, 2)
    Al *= (1 - 0.2 * tr)[..., None]
    E += hexc('#f4f8ff') * (tr * 0.3)[..., None]
    # wet specular corner lines (both front edges) + top lip
    ew = max(1, int(0.012 * w))
    A.rect(E, 0, 0, ew, h * 0.97, hexc('#ffc0e8') * 0.9)
    A.rect(E, w - ew, 0, w, h * 0.97, hexc('#c0ecff') * 0.9)
    A.rect(E, 0, 0, w, max(1, int(0.006 * h)), hexc('#ffffff') * 0.8)


# ----------------------------------------------------------------------------- foreground neon rain
def fg_rain_setup(sc):
    W, H = sc.W, sc.H
    rng = np.random.default_rng(4747)
    n = 44
    sc._fgr = dict(x=rng.uniform(-0.05 * W, 1.1 * W, n), ph=rng.uniform(0, 1, n),
                   L=rng.uniform(0.2, 0.42, n) * H, sp=rng.uniform(5.2, 6.6, n) * H,
                   wd=rng.uniform(1.6, 2.8, n) * H / 1080.0, a=rng.uniform(0.55, 1.0, n))


def fg_rain(sc, img, t, light):
    """Fewer, longer, brighter near streaks, clearly faster than the fine far rain; each takes the hue of the
    neon behind it (pink by the ラーメン signs, cyan by メン横丁). light: (H, W, 3) blurred scene light."""
    if not hasattr(sc, '_fgr'):
        fg_rain_setup(sc)
    W, H = sc.W, sc.H
    R_ = sc._fgr
    slant = 0.09
    m = np.zeros((H, W), np.uint8)
    for x, ph, L, sp, wd, a in zip(R_['x'], R_['ph'], R_['L'], R_['sp'], R_['wd'], R_['a']):
        per = H + L
        y = (ph * per + sp * t) % per - L
        xs = x - slant * (y + L)
        p0 = (int(xs * 4), int(y * 4))
        p1 = (int((xs - slant * L) * 4), int((y + L) * 4))
        cv2.line(m, p0, p1, int(255 * a), max(1, int(round(wd))), cv2.LINE_AA, 2)
    mf = m.astype(np.float32) / 255.0
    mf = cv2.GaussianBlur(mf, (0, 0), 0.6 * H / 1080.0)
    # hue from the local light, pushed toward its chroma so the streaks read as neon-tinted, plus a white core
    lum = light.max(-1, keepdims=True)
    chroma = light / np.maximum(lum, 1e-3)
    col = chroma * np.clip(lum * 2.2, 0.25, 1.6) * 0.75 + 0.1
    img += mf[..., None] * col


# ----------------------------------------------------------------------------- vanishing-point mist
def mist(sc, img, t):
    """A faint layer of low mist drifting slowly across the far street around the vanishing point."""
    W, H = sc.W, sc.H
    cam = sc.cam
    if not hasattr(sc, '_mist'):
        PW = int(W * 1.6)
        n1 = A.noise(max(PW // 6, 8), max(H // 24, 8), 9, 8181, 4)
        n1 = cv2.resize(n1, (PW, H), interpolation=cv2.INTER_CUBIC)
        n2 = A.noise(max(PW // 3, 8), max(H // 8, 8), 18, 8182, 3)
        n2 = cv2.resize(n2, (PW, H), interpolation=cv2.INTER_CUBIC)
        den = np.clip((n1 * 0.7 + n2 * 0.3 - 0.38) * 2.2, 0, 1) ** 1.3
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = cam.cx + 0.02 * W, cam.cy - 0.035 * H
        env = np.exp(-((xx - cx) / (0.16 * W)) ** 2 - ((yy - cy) / (0.075 * H)) ** 2)
        sc._mist = (den.astype(np.float32), env.astype(np.float32))
    den, env = sc._mist
    PW = den.shape[1]
    off = (0.25 * W + 0.035 * W * t)
    o = int(off)
    fr = np.float32(off - o)
    d = den[:, o:o + W] * (1 - fr) + den[:, o + 1:o + 1 + W] * fr
    a = (d * env * 0.5)[..., None]
    col = np.float32([0.52, 0.60, 0.60])
    img[:] = img * (1 - a * 0.55) + col * a


# ----------------------------------------------------------------------------- rain-haze halation
def rain_halo(sc, img, t):
    """Neon halation scattered in the rain: the brightest (saturated) sources bleed wide soft halos whose
    density is broken by slowly drifting vertical rain sheets."""
    W, H = sc.W, sc.H
    w4, h4 = W // 4, H // 4
    small = cv2.resize(img, (w4, h4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 1.15, 0, 2.0)
    g = cv2.GaussianBlur(br, (0, 0), 0.012 * W / 4) * 0.6 + cv2.GaussianBlur(br, (0, 0), 0.04 * W / 4) * 0.5
    if not hasattr(sc, '_sheet'):
        sh = A.noise(max(w4 // 2, 8), 6, max(8, w4 // 8), 9393, 3)
        sc._sheet = cv2.resize(sh, (w4 * 2, h4), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    off = 0.06 * w4 * t
    o = int(off)
    fr = np.float32(off - o)
    s = sc._sheet[:, w4 // 2 - o: w4 // 2 - o + w4] * (1 - fr) + sc._sheet[:, w4 // 2 - o - 1: w4 // 2 - o - 1 + w4] * fr
    g = g * (0.45 + 1.1 * np.clip(s - 0.3, 0, 1))[..., None]
    img += cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * 0.3


# ----------------------------------------------------------------------------- splash crowns on edges
def splash_setup(sc):
    """Rain impacts on the tops of signs, lanterns and stand signs: (X0, X1, Y, Z, rate/s)."""
    rng = np.random.default_rng(7373)
    surf = []
    for (side, z, y0, hh, w, st) in sc.signs:
        if z > 40:
            continue
        xw = sc._wall_x(side, z)
        gap = 0.25
        x0, x1 = (xw + gap, xw + gap + w) if side < 0 else (xw - gap - w, xw - gap)
        surf.append((x0, x1, y0 + hh + 0.01, z + 0.15, 30 * w))
    for (xw, z) in [(3.85, 15.0), (3.85, 17.8), (-3.35, 10.2), (-3.35, 12.4)]:
        X = xw - np.sign(xw) * 0.35
        surf.append((X - 0.12, X + 0.12, 2.28, z, 6))
    for (X, Z) in ((3.05, 9.2), (-2.75, 16.5)):
        surf.append((X - 0.22, X + 0.22, 1.13, Z, 14))
    ev = []
    for (x0, x1, Y, Z, rate) in surf:
        k = int(rate * 5.4)
        for _ in range(k):
            ev.append((rng.uniform(x0, x1), Y, Z, rng.uniform(-0.2, 5.2), rng.uniform(0, 2 * np.pi)))
    sc._spl = np.array(ev, np.float64)


def splashes(sc, img, t, dX, dY, dZ):
    if not hasattr(sc, '_spl'):
        splash_setup(sc)
    cam = sc.cam
    E = sc._spl
    fa = (t - E[:, 3]) * C.FPS
    act = (fa >= 0) & (fa < 2)
    if not act.any():
        return
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    for X, Y, Z, t0, jit in E[act]:
        zc = Z - dZ
        if zc < 0.5:
            continue
        x = cam.cx + cam.f * (X - dX) / zc
        y = cam.cy - cam.f * (Y - cam.h - dY) / zc
        if not (0 <= x < W and 0 <= y < H):
            continue
        sc._crown(m, x, y, cam.f * 0.026 / zc, (t - t0) * C.FPS, jit, 200)
    mf = m.astype(np.float32) / 255.0
    loc = cv2.resize(cv2.resize(img, (W // 16, H // 16), interpolation=cv2.INTER_AREA), (W, H))
    img += mf[..., None] * (0.3 + 0.5 * np.clip(loc, 0, 1.5))


# ----------------------------------------------------------------------------- walker foot splashes
def foot_splashes(sc, img, t, dX, dY, dZ, X, Z0, speed, cycle):
    """Small splash crowns + a soft spray puff where each foot plants (two plants per cycle seconds)."""
    cam = sc.cam
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    puff = []
    for side, lx, off in ((-1, -0.1, 0.0), (1, 0.1, 0.5)):
        # time since this foot last planted
        tp = math.floor((t / cycle) - off) * cycle + off * cycle
        age = (t - tp) * C.FPS
        if age < 0 or age >= 5:
            continue
        Zf = Z0 + speed * tp
        zc = Zf - dZ
        x = cam.cx + cam.f * (X + lx - dX) / zc
        y = cam.cy + cam.f * (cam.h + dY) / zc
        s = cam.f * 0.05 / zc
        if age < 2:
            sc._crown(m, x, y - s * 0.1, s, age, 1.3 + side, 255)
        puff.append((x, y - s * 0.4, s * (1.0 + 0.5 * age), max(0.0, 1 - age / 5)))
    img += (m.astype(np.float32) / 255.0)[..., None] * np.float32([0.95, 1.0, 1.1])
    for (x, y, r, a) in puff:
        r = max(r, 1.0)
        x0, y0 = int(x - 3 * r), int(y - 3 * r)
        x1, y1 = int(x + 3 * r) + 1, int(y + 2 * r) + 1
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if xb <= xa or yb <= ya:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        g = np.exp(-((xx - x) / (1.6 * r)) ** 2 - ((yy - y) / (0.8 * r)) ** 2) * a * 0.35
        img[ya:yb, xa:xb] += g[..., None] * np.float32([0.85, 0.9, 1.0])
