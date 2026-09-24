"""Round-8 painting helpers for s07_comet_night (Your Name comet night).

- comet_tail6: the Your-Name split comet. The main DUST tail is one broad, soft, curved fan (white-cyan
  core line on its leading edge, cyan -> blue body, ragged hair-like striae on both edges, flaring out and
  losing opacity toward the far end). Separate thin secondary bands run along its OUTER (trailing) edge:
  a magenta/violet band and a fainter, broken gold band, each with its own striae and dark sky gaps
  between them (no banded rainbow gradient, no green).
- ion_tail6: a thin, straight, hard-edged blue ion tail leaving the nucleus at a different angle.
"""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _g(x, c, w):
    return np.exp(-((x - c) / w) ** 2)


def _striae(u, vn, seed, fu=2.5, fv=40.0, blur_v=0.7, nv=1024, nu=48):
    """Lengthwise brush striae in 0..1: long along u, fine across vn."""
    rng = np.random.default_rng(seed)
    t = rng.random((nv, nu)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=1.3, sigmaY=blur_v)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    mx = np.mod(u * fu * nu / 8.0, nu).astype(np.float32)
    my = np.mod(vn * fv + nv / 2, nv).astype(np.float32)
    return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _crop_box(Wp, Hp, head, d, L, W, bend, w1, pad=0.25):
    """Bounding box of the tail's influence region (so we only compute inside it)."""
    pts = []
    for u in np.linspace(0, 1.05, 12):
        for v in (-2.2, 2.6):
            wu = w1 * W * max(u, 0.02) ** 0.75
            acr = v * wu + bend * L * u * u
            pts.append((head[0] + d[0] * u * L - d[1] * acr, head[1] + d[1] * u * L + d[0] * acr))
    pts = np.array(pts)
    x0, y0 = pts.min(0) - pad * 0.1 * W
    x1, y1 = pts.max(0) + pad * 0.1 * W
    return (int(max(x0, 0)), int(max(y0, 0)), int(min(x1, Wp)), int(min(y1, Hp)))


def comet_tail6(Wp, Hp, head, d, L, W, s, bend=0.16, w0=0.003, w1=0.12, seed=3, secondary=True, wexp=0.75):
    """Returns (rgb additive, u, vn, body) full plate size."""
    bx0, by0, bx1, by1 = _crop_box(Wp, Hp, head, d, L, W, bend, w1)
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    dx, dy = d
    px, py = xs - head[0], ys - head[1]
    u = (px * dx + py * dy) / L
    across = -px * dy + py * dx
    across = across - bend * L * np.clip(u, 0, None) ** 2
    uc = np.clip(u, 0, 1.2)
    wu = (w0 + (w1 - w0) * np.clip(u, 0, 1.2) ** wexp) * W
    vn = across / wu

    # along-tail envelope: bright near the nucleus, a long slow fade (the fan loses opacity as it flares)
    start = C.smoothstep(-0.004, 0.01, u)
    fall = start * np.clip(1.02 - u, 0, 1) ** 1.35 * (0.3 + 0.7 * np.exp(-uc / 0.3))
    op = C.smoothstep(0.015, 0.2, uc)                # secondary structure opens out away from the head

    st = _striae(u, vn, seed, fu=2.2, fv=46.0)
    st2 = _striae(u * 0.6 + 2.0, vn, seed + 1, fu=1.6, fv=19.0)
    stf = _striae(u * 1.4 + 5.0, vn, seed + 2, fu=3.0, fv=110.0, blur_v=0.6, nv=2048)   # fine hairs
    hair = 0.55 * st + 0.3 * st2 + 0.15 * stf                                           # 0..1

    # ---------------- main dust fan
    # hard leading edge (vn ~ -0.6) broken into hairs; trailing edge ragged around vn ~ 0.25
    lead_e = -0.62 + 0.1 * (st - 0.5) * op
    lead = C.smoothstep(lead_e - 0.06, lead_e + 0.02, vn)
    trail_e = 0.18 + 0.32 * (hair - 0.5) * (0.3 + op) + 0.12 * op
    trail = C.smoothstep(trail_e + 0.2, trail_e - 0.12, vn)
    fan = lead * trail
    stria = 1 + (0.25 + 1.5 * hair - 1) * (0.35 + 0.65 * op)       # streaky body
    q = np.clip((vn + 0.62) / 0.9, 0, 1)                         # 0 at the lead edge .. 1 at the trail edge
    c_edge = np.array([0.72, 0.97, 1.0], np.float32)
    c_mid = np.array([0.22, 0.78, 1.0], np.float32)
    c_out = np.array([0.18, 0.42, 1.0], np.float32)
    qq = q[..., None]
    col = np.where(qq < 0.35, c_edge + (c_mid - c_edge) * (qq / 0.35),
                   c_mid + (c_out - c_mid) * np.clip((qq - 0.35) / 0.65, 0, 1))
    body_i = (0.95 - 0.45 * q) * fan * stria * fall
    img = col * body_i[..., None] * 2.0

    # white-cyan core line hugging the leading edge (fading out along the tail)
    core = _g(vn, -0.54 + 0.03 * (st - 0.5), 0.05 + 0.04 * op) * (0.75 + 0.5 * st2)
    core_env = start * np.exp(-uc / 0.42) * (1.0 + 1.2 * np.exp(-uc / 0.06))
    cc = np.clip(uc / 0.5, 0, 1)[..., None]
    core_col = np.array([0.88, 0.99, 1.0], np.float32) * (1 - cc) + np.array([0.35, 0.88, 1.0], np.float32) * cc
    img += (core * core_env)[..., None] * core_col * 1.25

    # hair jets feathering OUT of the leading edge (thin blue filaments)
    jet_m = np.clip(stf * 1.6 - 0.75, 0, 1) ** 1.5 * np.clip(st * 1.4 - 0.3, 0, 1)
    jet_prof = np.exp(-np.clip(lead_e - vn, 0, None) / (0.1 + 0.25 * op)) * (vn < lead_e + 0.02)
    img += (jet_m * jet_prof * fall * op)[..., None] * np.array([0.25, 0.6, 1.0], np.float32) * 1.6

    # ---------------- separate secondary bands on the outer edge
    # each band: own centre wobble along the tail, own striae, broken into long sections
    def band(c0, w, seed_b, u_on, brk_amt, wob=0.08):
        cen = c0 + wob * P.fbm1d(uc.ravel(), 3.0, 2, seed_b).reshape(uc.shape) + 0.08 * op
        sb = _striae(u * 0.8 + seed_b, vn, seed_b, fu=2.0, fv=34.0)
        e = np.abs(vn - cen) / w - 0.55 * (sb - 0.5)
        prof = C.smoothstep(1.0, 0.35, e)
        brk = C.smoothstep(-0.2, 0.35, P.fbm1d(uc.ravel(), 7.0, 3, seed_b + 50).reshape(uc.shape))
        brk = 1 - brk_amt + brk_amt * brk
        on = C.smoothstep(u_on, u_on + 0.12, uc)
        return prof * brk * on * (0.35 + 1.1 * sb) * fall

    mag = band(0.62, 0.13, 211, 0.05, 0.45) * secondary
    img += mag[..., None] * np.array([0.95, 0.22, 0.85], np.float32) * 1.5
    vio = band(0.8, 0.1, 223, 0.12, 0.6, wob=0.1) * secondary
    img += vio[..., None] * np.array([0.5, 0.25, 1.0], np.float32) * 0.5
    gold = band(1.02, 0.07, 227, 0.16, 0.7, wob=0.12) * secondary
    img += gold[..., None] * np.array([1.0, 0.68, 0.26], np.float32) * 3.0
    # a faint magenta fringe right at the leading edge (Your-Name spectral fringe), thin and broken
    fr = _g(vn, lead_e - 0.08, 0.035) * C.smoothstep(0.1, 0.3, uc) * fall * (0.3 + 0.9 * st2)
    img += fr[..., None] * np.array([0.7, 0.3, 1.0], np.float32) * 0.18

    # soft overall veil (the tail glows into the sky), cool
    veil = np.exp(-(np.clip(vn - 0.1, -4, 4) / 1.3) ** 2) * fall
    img += veil[..., None] * np.array([0.03, 0.08, 0.17], np.float32)

    body = (np.exp(-(vn + 0.2) ** 2 / 0.4) * fall).astype(np.float32)

    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img
    U = np.full((Hp, Wp), 5.0, np.float32)
    V = np.full((Hp, Wp), 9.0, np.float32)
    B = np.zeros((Hp, Wp), np.float32)
    U[by0:by1, bx0:bx1] = u
    V[by0:by1, bx0:bx1] = vn
    B[by0:by1, bx0:bx1] = body
    return out, U, V, B


def tail_particles6(Wp, Hp, head, d, L, W, s, bend, w0, w1, n, seed):
    """Glittering dust grains scattered through the fan (mostly cyan/white, a few magenta on the outer
    bands)."""
    rng = np.random.default_rng(seed)
    uq = rng.uniform(0.03, 0.8, n) ** 1.15
    vq = rng.normal(0.0, 0.5, n)
    wu = (w0 + (w1 - w0) * uq ** 0.75) * W
    acr = vq * wu + bend * L * uq ** 2
    x = head[0] + d[0] * uq * L - d[1] * acr
    y = head[1] + d[1] * uq * L + d[0] * acr
    cols = np.array([[0.75, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.85, 1.0], [1.0, 0.5, 0.95]], np.float32)
    ci = np.where(vq > 0.5, 3, rng.integers(0, 3, n))
    m = (rng.random(n) ** 6 * 1.8 + 0.06) * np.clip(1 - uq, 0.1, 1) ** 0.8
    acc = np.zeros((Hp, Wp, 3), np.float32)
    ok = (x >= 0) & (x < Wp) & (y >= 0) & (y < Hp)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m[ok] * cols[ci[ok], c])
    return C.blur(acc, 0.5 * s + 0.2) * 3.0


def ion_tail6(Wp, Hp, head, d, L, W, s, seed=7):
    """Thin, straight, hard-edged blue ion tail: a crisp bright line with a narrow glow and two faint
    parallel streamers, fading out along its length."""
    ex = head[0] + d[0] * L
    ey = head[1] + d[1] * L
    pad = 0.03 * W
    bx0, bx1 = int(max(min(head[0], ex) - pad, 0)), int(min(max(head[0], ex) + pad, Wp))
    by0, by1 = int(max(min(head[1], ey) - pad, 0)), int(min(max(head[1], ey) + pad, Hp))
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    a = -px * d[1] + py * d[0]
    uc = np.clip(u, 0, 1)
    w = (0.55 + 1.6 * uc) * s + 0.35
    fall = C.smoothstep(-0.003, 0.008, u) * np.clip(1 - u, 0, 1) ** 1.6 * (0.45 + 0.55 * np.exp(-uc / 0.25))
    core = C.smoothstep(w * 1.3, w * 0.35, np.abs(a))                    # hard edged
    glow = np.exp(-(a / (w * 5.0)) ** 2) * 0.22 + np.exp(-(a / (w * 16.0)) ** 2) * 0.06
    rng = np.random.default_rng(seed)
    stream = np.zeros_like(a)
    for k in range(3):
        off = rng.uniform(2.5, 6.0) * w * (1 if k % 2 else -1) * (0.3 + uc)
        stream += C.smoothstep(w * 0.9, w * 0.2, np.abs(a - off)) * rng.uniform(0.18, 0.32) * \
            C.smoothstep(0.08, 0.3, uc)
    br = 0.8 + 0.2 * np.sin(u * 37.0 + 1.3) * np.sin(u * 13.0)
    k = (core * br + glow + stream) * fall
    img = k[..., None] * np.array([0.32, 0.56, 1.0], np.float32) * 1.25
    img += (core * fall * np.exp(-uc / 0.08))[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img
    return out


def twilight6(Wp, Hp, y_h, H, xs, ys, x_warm, seed=0):
    """Wide warm kataware-doki glow behind the ranges: gold/amber at the horizon -> salmon -> violet,
    rising ~0.3 H so that 8-12% of frame height of warm gradient shows ABOVE the central range.
    Returns (additive rgb, mix) where mix suppresses the navy base under the warm part."""
    h = np.clip(y_h - ys, 0, None) / H
    gx = np.exp(-((xs - x_warm) / (0.42 * Wp)) ** 2)
    gxb = 0.4 + 0.6 * gx
    n = P.fbm_lowres(Wp, Hp, 3, 3, seed, q=8)
    hh = h * (1 + 0.12 * (n - 0.5))
    gold = np.exp(-hh / 0.1) * gxb
    salmon = np.exp(-((hh - 0.13) / 0.09) ** 2) * gxb * (0.9 + 0.2 * n) + np.exp(-hh / 0.16) * 0.35 * gxb
    violet = np.exp(-hh / 0.32) * (0.55 + 0.45 * gx)
    # thin stratus bars lying in the glow, lit gold/pink from below
    bars = P.rot_fbm(Wp, Hp, -1.5, 14.0, 40.0, 4, seed + 5, warp_amt=0.02)
    bars = C.smoothstep(0.6, 0.74, bars) * (np.exp(-((hh - 0.19) / 0.03) ** 2) + 0.7 * np.exp(-((hh - 0.26) / 0.02) ** 2))
    bars *= gx
    below = (ys <= y_h + 2)
    out = (gold[..., None] * np.array([1.0, 0.58, 0.2], np.float32) * 0.95 +
           salmon[..., None] * np.array([0.95, 0.42, 0.34], np.float32) * 0.42 +
           violet[..., None] * np.array([0.26, 0.09, 0.32], np.float32) * 0.42 +
           bars[..., None] * np.array([1.0, 0.55, 0.45], np.float32) * 0.28) * below[..., None]
    mix = np.clip(gold * 0.95 + salmon * 0.5 + violet * 0.15, 0, 0.92) * below
    return out.astype(np.float32), mix.astype(np.float32)


def conifer_bands(rgb, fmask, top_ss, depth, ss, s, Wp, th, light_x, seed, pal, haze_col, haze=0.35,
                  crown_col=(0.66, 0.82, 1.05)):
    """Paint a forest region as stacked bands of conifer tops (a painter's treeline): each band is a
    serrated line of spire tips with snow-dusted crowns catching the comet light along its upper edge and
    a darker body below; upper (farther) bands are lighter and bluer (aerial perspective), lower (nearer)
    bands darker and more saturated. rgb is modified in place inside fmask."""
    from s07_comet_night_hills import clumped_bumps
    rng = np.random.default_rng(seed)
    Hl = rgb.shape[0]
    xs2 = np.arange(Wp * ss, dtype=np.float32) / ss
    t1 = top_ss.reshape(Wp, ss).mean(1)
    dmax = float(np.percentile(depth, 90))
    sp = th * 2.1
    nb = int(min(max(dmax / sp, 1), 10)) + 1
    xr = np.arange(Wp, dtype=np.float32)
    near = np.exp(-np.abs(xr - light_x) / (0.4 * Wp))[None, :]
    fm = np.clip(cv2.GaussianBlur(fmask, (0, 0), 0.8 * s + 0.3) * 1.2, 0, 1)
    fs = pal['forest_sh']
    fl = pal['forest_lit']
    fb = pal['forest_base']
    crown = np.asarray(crown_col, np.float32)
    haze_col = np.asarray(haze_col, np.float32)
    for k in range(nb):
        fk = k / max(nb - 1, 1)
        und = P.fbm1d(xs2 / Wp, 7.0, 2, seed + 13 * k) * 0.35 * sp
        base = np.repeat(t1, ss) + (k + 0.3) * sp + und
        tip, _ = clumped_bumps(xs2, Wp, th * (1.1 + 0.5 * fk), seed + 29 * k, gap=-0.85)
        line = base - tip
        y0 = int(max(line.min() - 2, 0))
        y1 = int(min(Hl, base.max() + 2 * sp + th))
        if y1 <= y0:
            continue
        ysl = np.arange(y0 * ss, y1 * ss, dtype=np.float32)[:, None] / ss
        m = np.clip((ysl - line[None, :]) * ss + 0.5, 0, 1)
        m = cv2.resize(m, (Wp, y1 - y0), interpolation=cv2.INTER_AREA)
        ln1 = line.reshape(Wp, ss).min(1)
        tp1 = tip.reshape(Wp, ss).max(1)
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        d = np.clip(yy - ln1[None, :], 0, None)
        # body: darker toward the bottom of the band (the band below overlaps it)
        vb = np.exp(-d / (0.8 * sp))
        body = fb + (fs - fb) * vb[..., None] * 0.9 + (fl - fs) * (vb ** 2)[..., None] * 0.6 * near[..., None]
        # snow-dusted crowns: a thin pale cap on the tips, strongest on tall tips, lit side
        tipness = np.clip(tp1 / (th + 1e-3), 0, 1)[None, :]
        cap = np.exp(-d / (0.22 * th + 0.5)) * (0.3 + 0.7 * tipness) * (0.5 + 0.5 * near)
        body = body + (crown - body) * np.clip(cap * 0.9, 0, 0.85)[..., None]
        # aerial perspective: upper (farther) bands paler and bluer
        hz = haze * (1 - fk) ** 1.2
        body = body * (1 - hz) + haze_col * hz
        a = m * fm[y0:y1]
        rgb[y0:y1] = rgb[y0:y1] * (1 - a[..., None]) + body * a[..., None]
    return rgb
