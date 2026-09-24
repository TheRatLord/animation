"""Painted prop textures for s02_railway_crossing (round 3).

Props are painted once as premultiplied RGBA rasters in their own face coordinates (metres -> texels), with
real gradients, grime, stickers, glass reflections, a dark 1px contour and a warm sun-side rim; per frame they
are blitted into the supersampled vector canvas with a perspective warp of the face quad (so they sort with
the rest of the hard-edged structures).  Also: a textured concrete utility-pole cylinder drawn by remap.
"""
import math

import numpy as np
import cv2

import s02_railway_crossing_paint as P

hx = P.hexc
sstep = P.sstep


# ------------------------------------------------------------------------------------------ raster painter

class Tex:
    """RGBA straight-colour painter in face metres (u right, v up), ppm texels per metre, ss supersampling for
    crisp anti-aliased shapes. Layers are composited immediately (straight alpha over)."""

    def __init__(self, wm, hm, ppm=260):
        self.wm, self.hm, self.ppm = wm, hm, ppm
        self.w, self.h = int(round(wm * ppm)), int(round(hm * ppm))
        self.rgb = np.zeros((self.h, self.w, 3), np.float32)
        self.a = np.zeros((self.h, self.w), np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.U = (xx + 0.5) / ppm
        self.V = hm - (yy + 0.5) / ppm

    def px(self, u, v):
        return (u * self.ppm, (self.hm - v) * self.ppm)

    def mask_poly(self, pts, ss=4):
        m = np.zeros((self.h * ss, self.w * ss), np.uint8)
        p = np.array([self.px(u, v) for (u, v) in pts], np.float64) * ss * 16
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 255, cv2.LINE_8, shift=4)
        return cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def mask_rect(self, u0, v0, u1, v1, r=0.0):
        """Anti-aliased (rounded) rectangle mask via signed distance."""
        cu, cv_ = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        hu, hv = 0.5 * abs(u1 - u0) - r, 0.5 * abs(v1 - v0) - r
        qx = np.abs(self.U - cu) - hu
        qy = np.abs(self.V - cv_) - hv
        d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
        return np.clip(0.5 - d * self.ppm, 0, 1)

    def mask_circle(self, cu, cv_, r):
        d = np.sqrt((self.U - cu) ** 2 + (self.V - cv_) ** 2) - r
        return np.clip(0.5 - d * self.ppm, 0, 1)

    def fill(self, m, col, a=1.0):
        col = np.asarray(col, np.float32)
        k = np.clip(m * a, 0, 1)
        if col.ndim == 1:
            col = col[None, None, :]
        self.rgb = self.rgb * (1 - k[..., None]) + col * k[..., None]
        self.a = self.a + (1 - self.a) * k

    def mul(self, m, f):
        f = np.asarray(f, np.float32)
        self.rgb = self.rgb * (1 - m[..., None]) + self.rgb * f * m[..., None]

    def outline(self, px_w=2.2, col=hx('#1c1e28'), amt=0.85):
        """Darker contour just inside the silhouette (reads as a 1px line at screen scale)."""
        inside = (self.a > 0.5).astype(np.uint8)
        d = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        k = np.clip(px_w - d + 0.5, 0, 1) * (self.a > 0.02) * amt
        self.rgb = self.rgb * (1 - k[..., None]) + np.asarray(col, np.float32) * k[..., None]

    def rim(self, side=(1.0, 1.0), px_w=3.0, col=hx('#ffe2a8'), amt=0.9):
        """Warm rim light on the sun side (right / top edges) of the silhouette, just inside the contour."""
        inside = (self.a > 0.5).astype(np.uint8)
        # directional distance: shift mask toward the sun and see what leaves the silhouette
        sx, sy = side
        k = np.zeros_like(self.a)
        for i in range(1, int(px_w * 2) + 3):
            M = np.float32([[1, 0, -sx * i], [0, 1, sy * i]])
            sh = cv2.warpAffine(inside, M, (self.w, self.h), borderValue=0)
            k = np.maximum(k, (inside > 0) & (sh == 0))
        # keep a thin band a couple of texels inside the outline
        d = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
        band = np.clip(1 - np.abs(d - (px_w * 0.5 + 2.4)) / (px_w * 0.6), 0, 1)
        k = cv2.GaussianBlur(k.astype(np.float32), (0, 0), 0.8) * band * amt
        self.rgb = self.rgb * (1 - k[..., None]) + np.asarray(col, np.float32) * k[..., None]

    def premult(self):
        return np.dstack([self.rgb * self.a[..., None], self.a]).astype(np.float32)


def _noise(shape, seed, scale):
    h, w = shape
    t = P.tile_noise(256, seed=seed, octaves=4, base=4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return P.sample_tile(t, xx / scale, yy / scale)


def grime(T, v_top, amt=0.5, col=hx('#6a5a48'), seed=3, streaks=True):
    """Dirt gradient rising from the base, broken by noise, plus a few vertical rain streaks."""
    n = _noise((T.h, T.w), seed, 0.06 * T.ppm)
    g = np.clip((v_top - T.V) / v_top, 0, 1) ** 1.6 * (0.6 + 0.8 * n) * amt
    if streaks:
        rng = np.random.default_rng(seed)
        s = np.zeros((T.h, T.w), np.float32)
        for _ in range(int(T.wm * 9)):
            u = rng.uniform(0, T.wm)
            v0 = rng.uniform(0.2, T.hm)
            L = rng.uniform(0.1, 0.5)
            wd = rng.uniform(0.004, 0.012)
            s = np.maximum(s, np.exp(-((T.U - u) / wd) ** 2) * sstep(v0, v0 - 0.03, T.V) * sstep(v0 - L, v0 - L * 0.4, T.V) * 0.35)
        g = np.clip(g + s * amt, 0, 1)
    g = g * (T.a > 0)
    T.rgb = T.rgb * (1 - g[..., None]) + np.asarray(col, np.float32) * T.rgb.mean(-1, keepdims=True) * 1.2 * g[..., None]


# ------------------------------------------------------------------------------------------ vending machine

def vending_front(wm=1.0, hm=1.83, ppm=280):
    """Front face of a drinks vending machine, in the shade of the high sun (cool fill), with a lit display,
    glass sky reflection, coin panel, dispenser, stickers, grime, contour and warm rim on the sun side."""
    T = Tex(wm, hm, ppm)
    U, V = T.U, T.V
    body = T.mask_rect(0, 0, wm, hm, r=0.02)
    # shaded white enamel, cooler & darker toward the bottom (sky fill from above)
    bc = hx('#d6dde8') + (hx('#aeb8c8') - hx('#d6dde8')) * sstep(hm, 0, V)[..., None]
    T.fill(body, bc)
    # blue header with a white wave logo and abstract lettering
    hd = T.mask_rect(0.0, hm - 0.28, wm, hm, r=0.02)
    hcol = hx('#3478d0') + (hx('#1d4c9c') - hx('#3478d0')) * sstep(hm, hm - 0.28, V)[..., None]
    T.fill(hd, hcol)
    wave = np.exp(-((V - (hm - 0.15 + 0.035 * np.sin(U * 9.0))) / 0.012) ** 2) * hd * (U > 0.06) * (U < wm - 0.06)
    T.fill(wave, hx('#f4f8ff'), 0.95)
    for i in range(5):
        T.fill(T.mask_rect(0.1 + i * 0.075, hm - 0.255, 0.15 + i * 0.075, hm - 0.215, r=0.005), hx('#e8f0ff'), 0.9)
    T.fill(T.mask_rect(0.06, hm - 0.285, wm - 0.06, hm - 0.28), hx('#9fc4f0'), 1.0)
    # display window frame + backlit interior
    wx0, wx1, wy0, wy1 = 0.06, wm - 0.06, 0.96, hm - 0.33
    T.fill(T.mask_rect(wx0 - 0.025, wy0 - 0.025, wx1 + 0.025, wy1 + 0.025, r=0.012), hx('#8a94a4'))
    T.fill(T.mask_rect(wx0 - 0.012, wy0 - 0.012, wx1 + 0.012, wy1 + 0.012, r=0.008), hx('#4a5262'))
    inner = T.mask_rect(wx0, wy0, wx1, wy1)
    T.fill(inner, hx('#fbfdff') + (hx('#e2ecf6') - hx('#fbfdff')) * sstep(wy1, wy0, V)[..., None])
    rng = np.random.default_rng(12)
    cols = [hx('#e2402e'), hx('#f29a2e'), hx('#3a9a4a'), hx('#2f6fd0'), hx('#8a5a36'), hx('#f4f0e8'),
            hx('#d4c23a'), hx('#1c2a4a'), hx('#e86aa0'), hx('#6ac0e0')]
    rows, per = 3, 7
    rh = (wy1 - wy0) / rows
    for r in range(rows):
        yb = wy0 + r * rh + 0.045
        yt = yb + rh * 0.62
        cw = (wx1 - wx0) / per
        for j in range(per):
            x0 = wx0 + j * cw + cw * 0.14
            x1 = x0 + cw * 0.72
            cc = cols[int(rng.integers(0, len(cols)))]
            can = T.mask_rect(x0, yb, x1, yt, r=0.008)
            shade_x = 0.75 + 0.35 * np.clip(1 - np.abs((U - (x0 + x1) * 0.5) / (x1 - x0) * 2 - 0.25), 0, 1)
            T.fill(can, cc * shade_x[..., None])
            band = can * (V > yb + (yt - yb) * 0.42) * (V < yb + (yt - yb) * 0.62)
            T.fill(band, hx('#f8f8f4'), 0.85)
            T.fill(can * (V > yt - 0.018), hx('#dfe4ea'))
            T.fill(T.mask_rect(x0 + (x1 - x0) * 0.62, yb + 0.01, x0 + (x1 - x0) * 0.78, yt - 0.02), hx('#ffffff'), 0.55)
            # lit selection button under each can
            btn = T.mask_rect(x0 + 0.004, yb - 0.035, x1 - 0.004, yb - 0.015, r=0.004)
            T.fill(btn, hx('#6ae0ff') if (r + j) % 3 else hx('#9cff9a'))
        # price strip (dark shelf edge)
        T.fill(T.mask_rect(wx0, wy0 + r * rh + 0.005, wx1, wy0 + r * rh + 0.012), hx('#2a3040'))
    # glass: strong diagonal sky reflection + specular streak (sun side, upper right)
    d = (U - wx0) * 0.9 + (V - wy0) * 1.0
    refl = sstep(0.02, 0.1, d - 0.3) * sstep(0.34, 0.24, d - 0.3)
    T.fill(inner * refl, hx('#b8d8f6'), 0.72)
    refl2 = sstep(0.0, 0.04, d - 0.95) * sstep(0.2, 0.12, d - 0.95)
    T.fill(inner * refl2, hx('#d6eaff'), 0.6)
    streak = np.exp(-((d - 0.78) / 0.018) ** 2) + 0.6 * np.exp(-((d - 0.84) / 0.008) ** 2)
    T.fill(inner * np.clip(streak, 0, 1), hx('#ffffff'), 0.85)
    T.fill(inner * sstep(wy1 - 0.08, wy1, V), hx('#9cc0e0'), 0.35)
    # glowing display strip (LED marquee) just under the window
    strip = T.mask_rect(wx0, wy0 - 0.08, wx1, wy0 - 0.045, r=0.006)
    T.fill(strip, hx('#1a2034'))
    led = strip * (np.sin(U * 260.0) > -0.2) * T.mask_rect(wx0 + 0.02, wy0 - 0.072, wx1 - 0.25, wy0 - 0.053)
    T.fill(led, hx('#ffb040'), 0.95)
    # right control panel: coin slot, return lever, bill acceptor, digital display
    cp = T.mask_rect(wm - 0.33, 0.5, wm - 0.06, 0.9, r=0.012)
    T.fill(cp, hx('#b8c0cc') + (hx('#8e98a8') - hx('#b8c0cc')) * sstep(0.9, 0.5, V)[..., None])
    T.fill(T.mask_rect(wm - 0.33, 0.885, wm - 0.06, 0.9, r=0.004), hx('#e6ecf4'), 0.9)
    T.fill(T.mask_rect(wm - 0.24, 0.76, wm - 0.14, 0.845, r=0.01), hx('#dfe4ea'))           # chrome coin bezel
    T.fill(T.mask_rect(wm - 0.196, 0.772, wm - 0.184, 0.833, r=0.004), hx('#0e1016'))       # coin slot
    T.fill(T.mask_rect(wm - 0.2, 0.828, wm - 0.18, 0.836), hx('#ffffff'), 0.6)
    T.fill(T.mask_circle(wm - 0.105, 0.8, 0.022), hx('#d2d8e0'))                          # return lever
    T.fill(T.mask_circle(wm - 0.105, 0.8, 0.012), hx('#6a7280'))
    T.fill(T.mask_rect(wm - 0.3, 0.6, wm - 0.1, 0.655, r=0.006), hx('#1c1f26'))            # bill slot
    T.fill(T.mask_rect(wm - 0.28, 0.622, wm - 0.12, 0.632), hx('#5aff8a'), 0.9)
    disp = T.mask_rect(wm - 0.3, 0.69, wm - 0.12, 0.735, r=0.004)
    T.fill(disp, hx('#140c0c'))
    T.fill(disp * (np.sin(U * 330) > 0.1) * (np.abs(V - 0.712) < 0.012), hx('#ff5040'), 0.95)
    T.fill(T.mask_rect(wm - 0.31, 0.52, wm - 0.08, 0.57, r=0.004), hx('#f4f4ee'), 0.9)       # instruction sticker
    for i in range(3):
        T.fill(T.mask_rect(wm - 0.3 + i * 0.07, 0.535, wm - 0.25 + i * 0.07, 0.555), hx('#3a4a8a'), 0.7)
    # left ad panel with a big sticker
    ap = T.mask_rect(0.06, 0.48, wm - 0.37, 0.9, r=0.012)
    T.fill(ap, hx('#eef3f8'))
    T.fill(T.mask_rect(0.08, 0.5, wm - 0.39, 0.64, r=0.008), hx('#2a6cc4'))
    T.fill(T.mask_circle(0.2, 0.76, 0.1) * ap, hx('#f2a23a'))
    T.fill(T.mask_circle(0.2, 0.76, 0.07) * ap, hx('#ffe07a'))
    T.fill(T.mask_rect(0.33, 0.72, wm - 0.4, 0.8, r=0.006), hx('#e04a3a'), 0.9)
    T.fill(T.mask_rect(0.1, 0.56, 0.5, 0.58), hx('#ffffff'), 0.9)
    # dispenser opening: dark recess with a lit lip
    T.fill(T.mask_rect(0.09, 0.1, wm - 0.12, 0.42, r=0.02), hx('#3c4250'))
    T.fill(T.mask_rect(0.12, 0.13, wm - 0.15, 0.37, r=0.012), hx('#1a1e28') + (hx('#2e3442') - hx('#1a1e28')) * sstep(0.13, 0.37, V)[..., None])
    T.fill(T.mask_rect(0.12, 0.35, wm - 0.15, 0.37), hx('#6a7282'), 0.9)
    T.fill(T.mask_rect(0.09, 0.415, wm - 0.12, 0.425), hx('#e8eef6'), 0.6)
    # small stickers
    T.fill(T.mask_rect(wm - 0.11, 0.2, wm - 0.03, 0.28, r=0.006), hx('#f6d23a'))
    T.fill(T.mask_poly([(wm - 0.09, 0.215), (wm - 0.05, 0.215), (wm - 0.07, 0.265)]), hx('#1a1a1a'), 0.8)
    T.fill(T.mask_circle(0.05, 0.3, 0.025), hx('#e24a3a'))
    # plinth
    T.fill(T.mask_rect(0, 0, wm, 0.08), hx('#2e333c'))
    grime(T, 0.45, 0.55, seed=5)
    T.outline(2.4)
    T.rim((1.0, 1.0), 3.2, hx('#ffe8b4'), 0.95)
    return T.premult()


def vending_side(dm=0.72, hm=1.83, ppm=200):
    """-X side of the machine (in shadow): big ad graphic, grime, contour."""
    T = Tex(dm, hm, ppm)
    U, V = T.U, T.V
    body = T.mask_rect(0, 0, dm, hm, r=0.01)
    T.fill(body, hx('#aab4c6') + (hx('#8a94a8') - hx('#aab4c6')) * sstep(hm, 0, V)[..., None])
    T.fill(T.mask_rect(0, hm - 0.28, dm, hm), hx('#23549e'))
    ad = T.mask_rect(0.06, 0.55, dm - 0.06, hm - 0.36, r=0.02)
    T.fill(ad, hx('#3f7fca') + (hx('#78b8ea') - hx('#3f7fca')) * sstep(0.55, hm - 0.36, V)[..., None])
    for i in range(4):
        T.fill(T.mask_rect(0.12, 0.7 + i * 0.12, dm - 0.12, 0.75 + i * 0.12, r=0.01) * ad, hx('#e8f2fc'), 0.8)
    T.fill(T.mask_rect(0, 0, dm, 0.08), hx('#262a32'))
    grime(T, 0.5, 0.6, seed=7)
    T.mul(body, np.array([0.78, 0.82, 0.92], np.float32))
    T.outline(2.0)
    return T.premult()


def bin_front(wm=0.42, hm=0.82, ppm=280):
    T = Tex(wm, hm, ppm)
    U, V = T.U, T.V
    body = T.mask_rect(0, 0, wm, hm, r=0.03)
    T.fill(body, hx('#3a7ed0') + (hx('#28589c') - hx('#3a7ed0')) * sstep(hm, 0, V)[..., None])
    # moulded vertical ribs catching light on their right faces
    rib = (np.sin(U / wm * np.pi * 7) > 0.75) * body * (V > 0.08) * (V < hm - 0.2)
    T.fill(rib.astype(np.float32), hx('#5c9ae0'), 0.6)
    # can hole + label
    T.fill(T.mask_circle(wm / 2, hm - 0.16, 0.075), hx('#9cc4f0'))
    T.fill(T.mask_circle(wm / 2, hm - 0.16, 0.058), hx('#0c1422'))
    T.fill(T.mask_rect(0.07, 0.3, wm - 0.07, 0.48, r=0.01), hx('#f4f6f2'))
    for i in range(3):
        a0 = i * 2.094
        T.fill(T.mask_poly([(wm / 2 + 0.05 * math.cos(a0), 0.39 + 0.05 * math.sin(a0)),
                            (wm / 2 + 0.05 * math.cos(a0 + 0.9), 0.39 + 0.05 * math.sin(a0 + 0.9)),
                            (wm / 2 + 0.03 * math.cos(a0 + 0.5), 0.39 + 0.03 * math.sin(a0 + 0.5))]), hx('#2a8a4a'))
    T.fill(T.mask_rect(0.02, hm - 0.05, wm - 0.02, hm - 0.035), hx('#a8d0ff'), 0.8)
    grime(T, 0.3, 0.5, seed=9, streaks=False)
    T.outline(2.4)
    T.rim((1.0, 1.0), 3.0, hx('#ffe2a8'), 0.9)
    return T.premult()


def gearbox_front(wm=0.38, hm=1.0, ppm=280, mirror=False):
    """Barrier machine housing front: diagonal yellow/black hazard stripes, painted metal with fall-off,
    access-door seam, vent louvres, maker plate, grime and a warm rim on the sun side."""
    T = Tex(wm, hm, ppm)
    U, V = T.U, T.V
    body = T.mask_rect(0, 0, wm, hm, r=0.012)
    s = 1.0 if not mirror else -1.0
    ph = (V * 1.0 + U * 0.55 * s) / 0.26
    stripe = np.clip((np.abs((ph % 1.0) - 0.5) - 0.25) * T.ppm * 0.26 + 0.5, 0, 1)
    yel = hx('#ffc81e') * 0.82
    blk = hx('#22242c')
    col = yel[None, None, :] * (1 - stripe[..., None]) + blk[None, None, :] * stripe[..., None]
    # cool fill in shade, lighter toward the top (sky), subtle vertical fall-off
    col = col * (0.72 + 0.28 * sstep(0, hm, V))[..., None] * np.array([0.92, 0.95, 1.08], np.float32)
    T.fill(body, col)
    # door seam, hinges, louvres, plate
    T.fill(T.mask_rect(0.04, 0.12, wm - 0.04, 0.125), hx('#101218'), 0.7)
    T.fill(T.mask_rect(0.04, 0.12, 0.045, hm - 0.1), hx('#101218'), 0.6)
    T.fill(T.mask_rect(wm - 0.045, 0.12, wm - 0.04, hm - 0.1), hx('#101218'), 0.6)
    T.fill(T.mask_rect(0.04, hm - 0.105, wm - 0.04, hm - 0.1), hx('#101218'), 0.6)
    for i in range(4):
        v0 = hm - 0.2 - i * 0.035
        T.fill(T.mask_rect(0.1, v0, wm - 0.1, v0 + 0.014, r=0.004), hx('#16181e'), 0.9)
        T.fill(T.mask_rect(0.1, v0 + 0.014, wm - 0.1, v0 + 0.019), hx('#d8dce4'), 0.6)
    T.fill(T.mask_rect(0.1, 0.5, wm - 0.1, 0.6, r=0.006), hx('#e8e8e0'))
    T.fill(T.mask_rect(0.12, 0.56, wm - 0.12, 0.575), hx('#3a4a7a'), 0.8)
    T.fill(T.mask_rect(0.12, 0.525, wm - 0.18, 0.54), hx('#3a4a7a'), 0.6)
    T.fill(T.mask_circle(wm / 2, 0.3, 0.016), hx('#c8ccd4'))       # key lock
    T.fill(T.mask_circle(wm / 2, 0.3, 0.007), hx('#1a1a20'))
    T.fill(T.mask_rect(0, 0, wm, 0.06), hx('#1a1c22'))
    grime(T, 0.4, 0.6, seed=11 + int(mirror))
    T.outline(2.4)
    T.rim((1.0, 1.0), 3.0, hx('#fff0b8'), 0.95)
    return T.premult()


def pole_texture(Hm, seed=0, h=900, w=72):
    """Concrete pole unrolled: columns = cos-angle across the silhouette (-1 left .. +1 right), rows = height.
    Cel-banded cylindrical shading with warm lit side (right), cool blue shadow side, a hot rim highlight,
    concrete mottling, form seams, rain stains and base grime. Returns straight RGB (h, w, 3)."""
    rng = np.random.default_rng(seed)
    u = (np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1
    v = (np.arange(h, dtype=np.float32)[::-1] + 0.5) / h * Hm        # row 0 = top
    UU, VV = np.meshgrid(u, v)
    # normal: x = u, z = -sqrt(1-u^2) (toward camera); sun L = (0.45, 0.72, 0.53)
    L = np.array([0.72, 0.6, 0.36])
    L = L / np.linalg.norm(L)
    nz = -np.sqrt(np.clip(1 - UU ** 2, 0, 1))
    ndl = UU * L[0] + nz * L[2]
    lit = hx('#ffe8c8')
    mid = hx('#9ea2c0')
    shd = hx('#5a6694')
    deep = hx('#3c4674')
    k1 = sstep(-0.06, -0.02, ndl)
    k2 = sstep(0.12, 0.16, ndl)
    col = deep + (shd - deep) * sstep(-0.9, -0.5, ndl)[..., None]
    col = col + (mid - col) * k1[..., None]
    col = col + (lit - col) * k2[..., None]
    # reflected warm bounce from the ground on the far-left edge, hot rim on the right edge
    col = col + (hx('#a898a8') - col) * (sstep(-0.75, -0.98, UU) * 0.5)[..., None]
    col = col + (hx('#fffcf4') - col) * (sstep(0.84, 0.9, UU) * 0.95)[..., None]
    # concrete mottling & pores
    n1 = _noise((h, w), seed + 1, 18.0)
    n2 = _noise((h, w), seed + 2, 3.0)
    col = col * (0.93 + 0.12 * n1 + 0.06 * (n2 - 0.5))[..., None]
    pores = (n2 < 0.12).astype(np.float32) * 0.25
    col = col * (1 - pores[..., None])
    # rain stains streaming down from the cross-arm + seams
    st = np.zeros((h, w), np.float32)
    for _ in range(7):
        uc = rng.uniform(-0.8, 0.8)
        v0 = rng.uniform(Hm * 0.4, Hm * 0.95)
        L_ = rng.uniform(1.0, 3.5)
        st = np.maximum(st, np.exp(-((UU - uc) / rng.uniform(0.04, 0.1)) ** 2) * sstep(v0, v0 - 0.2, VV) *
                        sstep(v0 - L_, v0 - L_ * 0.5, VV))
    col = col * (1 - st[..., None] * np.array([0.22, 0.24, 0.2], np.float32))
    for hs in (3.0, 6.0, 9.0):
        if hs < Hm - 1:
            col = col * (1 - 0.35 * np.exp(-((VV - hs) / 0.015) ** 2))[..., None]
    g = np.clip((1.8 - VV) / 1.8, 0, 1) ** 1.5 * (0.5 + 0.6 * n1)
    col = col + (col * np.array([0.72, 0.64, 0.52], np.float32) - col) * g[..., None]
    return col.astype(np.float32)


# ------------------------------------------------------------------------------------------ blitting

def blit_quad(cv, tex_pm, quad):
    """Warp a premultiplied RGBA texture onto a screen quad (TL, TR, BR, BL in 1x screen px) and composite
    it over the supersampled canvas buffer."""
    ss = cv.ss
    q = np.asarray(quad, np.float32) * ss
    x0, y0 = np.floor(q.min(0)).astype(int) - 1
    x1, y1 = np.ceil(q.max(0)).astype(int) + 2
    Hb, Wb = cv.buf.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, Wb), min(y1, Hb)
    if X1 <= X0 or Y1 <= Y0:
        return
    th, tw = tex_pm.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, q - np.float32([X0, Y0]))
    scale = math.sqrt(abs(np.linalg.det(M[:2, :2]))) if M[2, 0] == 0 else None
    t = tex_pm
    # pre-filter when minifying strongly
    dw = (q[:, 0].max() - q[:, 0].min()) / tw
    if dw < 0.5:
        fct = max(dw * 1.6, 0.05)
        t = cv2.resize(tex_pm, (max(int(tw * fct), 2), max(int(th * fct), 2)), interpolation=cv2.INTER_AREA)
        src2 = np.float32([[0, 0], [t.shape[1], 0], [t.shape[1], t.shape[0]], [0, t.shape[0]]])
        M = cv2.getPerspectiveTransform(src2, q - np.float32([X0, Y0]))
    w = cv2.warpPerspective(t, M, (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    region = cv.buf[Y0:Y1, X0:X1].astype(np.float32)
    a = w[..., 3:4]
    region = region * (1 - a) + w * 255.0
    cv.buf[Y0:Y1, X0:X1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


def blit_pole(cv, tex, xc, yb, yt, rb, rt, fog=0.0, hazec=None):
    """Draw a vertical tapered cylinder (screen x centre xc, base yb, top yt, radii rb/rt px) textured with
    pole_texture() into the canvas by remapping each screen pixel to (angle, height)."""
    ss = cv.ss
    Hb, Wb = cv.buf.shape[:2]
    r_max = max(rb, rt)
    X0, X1 = max(int((xc - r_max - 2) * ss), 0), min(int((xc + r_max + 2) * ss) + 1, Wb)
    Y0, Y1 = max(int(yt * ss) - 1, 0), min(int(yb * ss) + 2, Hb)
    if X1 <= X0 or Y1 <= Y0:
        return
    xs = (np.arange(X0, X1, dtype=np.float32) + 0.5) / ss
    ys = (np.arange(Y0, Y1, dtype=np.float32) + 0.5) / ss
    fy = np.clip((yb - ys) / max(yb - yt, 1e-3), 0, 1)                      # 0 base .. 1 top
    r = rb + (rt - rb) * fy
    uu = (xs[None, :] - xc) / r[:, None]
    th, tw = tex.shape[:2]
    mx = ((uu + 1) * 0.5 * tw - 0.5).astype(np.float32)
    my = np.broadcast_to(((1 - fy) * (th - 1))[:, None], uu.shape).astype(np.float32)
    col = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if fog > 0:
        col = col * (1 - fog) + np.asarray(hazec, np.float32) * fog
    ea = np.clip((1 - np.abs(uu)) * r[:, None] * ss * 0.5 + 0.5, 0, 1)
    ea = ea * ((ys >= yt) & (ys <= yb))[:, None]
    region = cv.buf[Y0:Y1, X0:X1].astype(np.float32)
    region = region * (1 - ea[..., None]) + np.dstack([col * 255.0, np.full(col.shape[:2], 255.0, np.float32)]) * ea[..., None]
    cv.buf[Y0:Y1, X0:X1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)
