"""Painted texture helpers for s08_snow_station: text boards (real kana), house windows with curtains and
interior, the vending machine front, a generic perspective quad paster and the foreground-fir repaint
(rim light, ambient fill, softened heavy snow clumps with blue undersides)."""
import numpy as np
import cv2

FONT_DIR = 'C:/Windows/Fonts'


def _font(size, bold=True):
    from PIL import ImageFont
    names = ('YuGothB.ttc', 'meiryob.ttc', 'msgothic.ttc') if bold else ('YuGothM.ttc', 'meiryo.ttc', 'msgothic.ttc')
    for n in names:
        try:
            return ImageFont.truetype(f'{FONT_DIR}/{n}', max(int(size), 6))
        except OSError:
            continue
    from PIL import ImageFont as IF
    return IF.load_default()


def text_board(w, h, lines, bg=(232, 232, 226), border=(40, 40, 46), vertical=False):
    """A small enamel sign: lines = [(text, rel_y, rel_size, rgb, bold)]; returns float RGB (h, w, 3)."""
    from PIL import Image, ImageDraw
    S = 3
    W, H = max(w, 4) * S, max(h, 4) * S
    im = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(im)
    bw = max(int(min(W, H) * 0.06), 1)
    d.rectangle([0, 0, W - 1, H - 1], outline=border, width=bw)
    for (txt, ry, rs, col, bold) in lines:
        d.text((W / 2, H * ry), txt, font=_font(H * rs, bold), fill=col, anchor='mm')
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (max(w, 4), max(h, 4)), interpolation=cv2.INTER_AREA)


def timetable(w, h):
    """Station timetable board: title bar, hour column, rows of minute figures."""
    from PIL import Image, ImageDraw
    S = 3
    W, H = max(w, 8) * S, max(h, 8) * S
    im = Image.new('RGB', (W, H), (238, 238, 232))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, int(H * 0.12)], fill=(30, 70, 150))
    d.text((W / 2, H * 0.06), '時刻表', font=_font(H * 0.08), fill=(250, 250, 250), anchor='mm')
    rows = 11
    rng = np.random.default_rng(3)
    for r in range(rows):
        y = H * (0.16 + r * 0.075)
        d.line([(0, y + H * 0.068), (W, y + H * 0.068)], fill=(170, 170, 170), width=max(S // 2, 1))
        d.text((W * 0.1, y + H * 0.034), str(6 + r * 1 + (r > 6) * 2), font=_font(H * 0.05), fill=(20, 20, 30), anchor='mm')
        mins = sorted(rng.choice(60, rng.integers(0, 3) + 1, replace=False))
        for k, m in enumerate(mins):
            d.text((W * (0.35 + k * 0.22), y + H * 0.034), f'{m:02d}', font=_font(H * 0.045, False),
                   fill=(180, 30, 30) if (r + k) % 4 == 0 else (30, 30, 40), anchor='mm')
    d.line([(W * 0.2, H * 0.12), (W * 0.2, H)], fill=(120, 120, 120), width=S)
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (max(w, 8), max(h, 8)), interpolation=cv2.INTER_AREA)


def poster(w, h, seed=0):
    """Travel poster: gradient scene (sea / mountain / sunset) with a title band of kana."""
    from PIL import Image, ImageDraw
    S = 3
    W, H = max(w, 8) * S, max(h, 8) * S
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, H)[:, None, None]
    pal = [((0.95, 0.55, 0.4), (0.3, 0.55, 0.9)), ((0.35, 0.7, 0.95), (0.95, 0.95, 0.85)), ((0.95, 0.75, 0.5), (0.4, 0.3, 0.6))][seed % 3]
    g = np.array(pal[0])[None, None] * (1 - yy) + np.array(pal[1])[None, None] * yy
    g = np.repeat(g, W, 1)
    # a mountain / island silhouette
    xs = np.linspace(0, 1, W)
    ridge = 0.62 - 0.18 * np.exp(-((xs - rng.uniform(0.3, 0.7)) / 0.18) ** 2) + 0.02 * np.sin(xs * 17)
    mm = (np.linspace(0, 1, H)[:, None] > ridge[None]).astype(np.float32)[..., None]
    g = g * (1 - mm) + np.array([0.18, 0.25, 0.45]) * mm
    im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(H * 0.8), W, H], fill=(245, 245, 240))
    title = ['ゆきまつり', '海へ', 'はるの旅'][seed % 3]
    d.text((W / 2, H * 0.9), title, font=_font(H * 0.1), fill=(30, 30, 40), anchor='mm')
    d.text((W / 2, H * 0.15), ['2月', 'JR', '北国'][seed % 3], font=_font(H * 0.09), fill=(255, 255, 255), anchor='mm')
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (max(w, 8), max(h, 8)), interpolation=cv2.INTER_AREA)


def route_map(w, h):
    """Line map board: white board, coloured route line with station dots and names."""
    from PIL import Image, ImageDraw
    S = 3
    W, H = max(w, 8) * S, max(h, 8) * S
    im = Image.new('RGB', (W, H), (236, 238, 234))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, int(H * 0.16)], fill=(40, 110, 60))
    d.text((W / 2, H * 0.08), '路線図', font=_font(H * 0.1), fill=(255, 255, 255), anchor='mm')
    pts = [(W * (0.08 + 0.84 * i / 6), H * (0.5 + 0.18 * np.sin(i * 1.3))) for i in range(7)]
    d.line(pts, fill=(38, 110, 190), width=max(int(H * 0.04), 2))
    for i, p in enumerate(pts):
        r = H * 0.035
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 255, 255), outline=(20, 20, 20), width=max(S // 2, 1))
        d.text((p[0], p[1] + H * 0.14), '駅', font=_font(H * 0.07, False), fill=(40, 40, 40), anchor='mm')
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (max(w, 8), max(h, 8)), interpolation=cv2.INTER_AREA)


def house_window(w, h, seed, bright=1.0):
    """A lit house window seen from outside at night: warm interior gradient (hot spot under a hanging
    lamp), drawn curtains on both sides with folds, a frosted lower edge, a sash cross, snow on the sill.
    Returns float RGB (h, w, 3) in HDR (emissive values > 1 allowed)."""
    w, h = max(int(w), 3), max(int(h), 3)
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    lx = rng.uniform(0.35, 0.65)
    warm_hi = np.array([1.45, 1.0, 0.55], np.float32)
    warm_lo = np.array([0.75, 0.38, 0.16], np.float32)
    hot = np.exp(-(((x - lx) / 0.35) ** 2 + ((y - 0.25) / 0.5) ** 2))[..., None]
    col = warm_lo + (warm_hi - warm_lo) * hot
    # interior: a dark horizontal shelf/sofa band and a back-wall picture
    if rng.random() < 0.7:
        band = ((y > 0.68) & (y < 0.8)).astype(np.float32)[..., None] * np.ones_like(x)[..., None]
        col = col * (1 - 0.45 * band)
    # curtains: a panel on each side with vertical folds, pulled open
    cl = rng.uniform(0.14, 0.3)
    cr = rng.uniform(0.14, 0.3)
    cur_col = np.array([1.1, 0.62, 0.32], np.float32) if rng.random() < 0.6 else np.array([0.95, 0.75, 0.55], np.float32)
    left = x < cl + 0.04 * np.sin(y * 6)
    right = x > 1 - cr - 0.04 * np.sin(y * 6 + 1)
    folds = 0.8 + 0.2 * np.cos(x * w * 0.9 * np.pi / max(w * 0.06, 1))
    cm = (left | right)[..., None].astype(np.float32)
    col = col * (1 - cm) + (cur_col * folds[..., None] * (0.75 + 0.3 * hot)) * cm * bright ** 0.2
    # pendant lamp: small dark shade with a bright bulb, just under the top
    if rng.random() < 0.6 and w > 6:
        d = np.sqrt(((x - lx) / 0.1) ** 2 + ((y - 0.16) / 0.07) ** 2)
        shade = (d < 1) & (y < 0.18)
        col = np.where(shade[..., None], np.array([0.25, 0.12, 0.05], np.float32), col)
        bulb = np.exp(-(((x - lx) / 0.05) ** 2 + ((y - 0.2) / 0.04) ** 2))[..., None]
        col = col + bulb * np.array([2.5, 2.0, 1.3], np.float32)
    # sash cross
    mw = max(0.5 / w, 0.03)
    mh = max(0.5 / h, 0.03)
    sash = (np.abs(x - 0.5) < mw) | (np.abs(y - 0.5) < mh)
    col = np.where(sash[..., None], np.array([0.12, 0.07, 0.05], np.float32), col)
    # frost creeping up from the lower corners
    fr = np.clip(1 - (1 - y) / 0.28 - 0.35 * np.minimum(x, 1 - x) / 0.5, 0, 1) ** 1.5
    fr = fr + np.clip(1 - np.minimum(x, 1 - x) / 0.08 - (1 - y) / 0.6, 0, 1) * 0.6
    col = col * (1 - 0.5 * fr[..., None]) + fr[..., None] * np.array([0.85, 0.85, 0.95], np.float32) * 0.6
    # the glass reflects the snowy night sky: a cool sheen across the upper panes with one diagonal glint
    refl = np.clip(1 - y / 0.45, 0, 1) ** 1.5 * 0.12 + 0.1 * np.exp(-((x + y * 0.6 - 0.55) / 0.06) ** 2) * (y < 0.5)
    col = col + refl[..., None] * np.array([0.35, 0.45, 0.8], np.float32) / max(bright, 0.5)
    return (col * bright).astype(np.float32)


def vending_front(tw=72, th=144, seed=4):
    """Japanese drink vending machine front: red body, lit display window with three shelves of bottles
    and cans (each with its own colour and a white price tag and a tiny button lamp), a lit logo panel,
    a coin panel and the dark pick-up slot. HDR float RGB (th, tw, 3)."""
    rng = np.random.default_rng(seed)
    red = np.array([0.55, 0.05, 0.06], np.float32)
    tex = np.ones((th, tw, 3), np.float32) * red
    # lit display window (warm-white)
    y0, y1 = int(th * 0.06), int(th * 0.56)
    x0, x1 = int(tw * 0.07), int(tw * 0.93)
    yy = np.linspace(0, 1, y1 - y0)[:, None, None]
    tex[y0:y1, x0:x1] = np.array([1.35, 1.35, 1.25], np.float32) * (1.05 - 0.15 * yy)
    rows = 3
    rh = (y1 - y0) / rows
    cols_ = [[1.6, 0.25, 0.2], [0.25, 0.55, 1.6], [1.6, 1.2, 0.2], [0.3, 1.3, 0.45], [1.5, 1.45, 1.35],
             [1.1, 0.45, 1.1], [0.2, 0.2, 0.25], [1.5, 0.75, 0.2]]
    n = 6
    cw = (x1 - x0) / n
    for r in range(rows):
        ry = y0 + int(r * rh)
        # shelf shadow
        tex[ry + int(rh * 0.82): ry + int(rh), x0:x1] *= 0.55
        for k in range(n):
            c = np.array(cols_[rng.integers(len(cols_))], np.float32)
            bx0 = int(x0 + k * cw + cw * 0.18)
            bx1 = int(x0 + (k + 1) * cw - cw * 0.18)
            by0 = ry + int(rh * (0.12 if rng.random() < 0.5 else 0.25))
            by1 = ry + int(rh * 0.66)
            tex[by0:by1, bx0:bx1] = c
            tex[by0:by0 + max(1, int(rh * 0.08)), bx0:bx1] = c * 0.5 + 0.5     # cap/lid highlight
            tex[ry + int(rh * 0.7): ry + int(rh * 0.8), bx0:bx1] = [1.5, 1.5, 1.5]   # price tag
            bl = int(x0 + (k + 0.5) * cw)
            tex[ry + int(rh * 0.7): ry + int(rh * 0.8), bl: bl + 1] = [0.3, 1.8, 0.4] if rng.random() < 0.8 else [1.8, 0.3, 0.2]
    # logo band
    ly0, ly1 = int(th * 0.59), int(th * 0.68)
    tex[ly0:ly1, x0:x1] = [1.5, 1.45, 1.35]
    tex[ly0 + 2:ly1 - 2, int(tw * 0.25):int(tw * 0.75)] = [1.4, 0.2, 0.15]
    # coin panel + slot
    tex[int(th * 0.7):int(th * 0.8), int(tw * 0.62):int(tw * 0.85)] = [0.2, 0.2, 0.22]
    tex[int(th * 0.72):int(th * 0.75), int(tw * 0.66):int(tw * 0.8)] = [0.9, 1.2, 0.6]
    tex[int(th * 0.84):int(th * 0.94), int(tw * 0.12):int(tw * 0.88)] = [0.02, 0.02, 0.025]
    tex[int(th * 0.84):int(th * 0.85), int(tw * 0.12):int(tw * 0.88)] = [0.5, 0.15, 0.1]
    return tex


def paste_quad(cv, pts, tex, z, alpha=1.0):
    """Perspective-map a texture (h, w, 3) onto the screen quad pts (bl, br, tr, tl) on canvas cv."""
    th, tw = tex.shape[:2]
    src = np.float32([[0, th], [tw, th], [tw, 0], [0, 0]])
    dst = np.float32(pts)
    M = cv2.getPerspectiveTransform(src, dst)
    x0 = int(np.floor(dst[:, 0].min())) - 2
    y0 = int(np.floor(dst[:, 1].min())) - 2
    w = int(np.ceil(dst[:, 0].max())) - x0 + 3
    h = int(np.ceil(dst[:, 1].max())) - y0 + 3
    if w < 1 or h < 1:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ M
    img = cv2.warpPerspective(tex.astype(np.float32), T, (w, h), flags=cv2.INTER_LINEAR)
    # 3x supersampled coverage for anti-aliased edges
    S = 3
    T3 = np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]], np.float64) @ T
    a = cv2.warpPerspective(np.ones((th, tw), np.float32), T3, (w * S, h * S), flags=cv2.INTER_NEAREST)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    cv.paint(x0, y0, a * alpha, img, z)


def repaint_fir(rgb, a, s, lamp_uv=None, sky_dir=(-0.55, -0.85)):
    """Repaint a foreground fir canvas in place (plate-space arrays): softened, heavier snow clumps with
    blue shadow undersides and a lit top, ambient sky fill in the needles, a cool blue rim on the
    sky-facing silhouette and a warm rim on edges facing the lamp."""
    H, W = a.shape
    rgb = rgb / np.maximum(a, 1e-3)[..., None]
    lum = rgb.mean(-1)
    inside = a > 0.02
    # snow = bright, bluish pixels inside the tree
    snow = np.clip((lum - 0.16) / 0.12, 0, 1) * (a > 0.3)
    # soften the zig-zag cut-outs: blur and re-threshold (rounded, heavier clumps)
    sig = 6.0 * s + 1.0
    sb = cv2.GaussianBlur(snow.astype(np.float32), (0, 0), sig)
    soft = np.clip((sb - 0.3) / 0.28, 0, 1)
    soft = soft * soft * (3 - 2 * soft)
    # vertical position inside each clump: top (lit) vs underside (blue shadow)
    sh = max(int(round(3.5 * s)), 2)
    up = np.zeros_like(soft)
    up[sh:] = soft[:-sh]           # value of the pixel `sh` above
    dn = np.zeros_like(soft)
    dn[:-sh] = soft[sh:]           # value of the pixel `sh` below
    top_edge = np.clip(soft - up, 0, 1)
    bot_zone = np.clip(soft - dn, 0, 1)
    bot_zone = cv2.GaussianBlur(bot_zone, (0, 0), 2.0 * s + 0.5)
    under = np.clip(soft - cv2.GaussianBlur(dn, (0, 0), 3 * s + 1), 0, 1)
    # base snow colour: take the existing snow colour but smoothed (no hard facets)
    snow_col = cv2.GaussianBlur(rgb * snow[..., None], (0, 0), sig) / np.maximum(sb, 1e-3)[..., None]
    snow_col = np.where((sb > 0.02)[..., None], snow_col, np.array([0.45, 0.55, 0.85], np.float32))
    shade_c = np.array([0.13, 0.18, 0.42], np.float32)
    top_c = np.array([0.78, 0.86, 1.08], np.float32)
    sc = snow_col * (1 - 0.55 * under[..., None]) + shade_c * (0.55 * under[..., None])
    sc = sc + top_edge[..., None] * (top_c - sc) * 0.55
    # needle mass: ambient sky fill (dark teal-blue instead of near-black), darker right under the snow
    fol = rgb.copy()
    # needle texture: short diagonal strokes (directionally blurred noise) in deep teal-blue, plus a
    # cool bounce from the snow ground on the undersides of each branch mass
    rng = np.random.default_rng(7)
    nz = rng.random((H, W)).astype(np.float32)
    k = max(int(9 * s) | 1, 3)
    ker = np.zeros((k, k), np.float32)
    for i in range(k):
        ker[i, min(int(i * 0.6 + k * 0.2), k - 1)] = 1.0
    ker /= ker.sum()
    strokes = cv2.filter2D(nz, -1, ker)
    strokes = cv2.GaussianBlur(strokes, (0, 0), 0.6 * s + 0.3)
    strokes = np.clip((strokes - strokes.mean()) / (strokes.std() + 1e-6), -2, 2)
    ab_ = (a > 0.5).astype(np.float32)
    upv = np.zeros_like(ab_)
    upv[max(int(6 * s), 2):] = ab_[:-max(int(6 * s), 2)]
    underside = cv2.GaussianBlur(np.clip(ab_ - np.roll(ab_, -max(int(5 * s), 2), 0), 0, 1), (0, 0), 3 * s + 1)
    fol = fol * 0.95 + np.array([0.03, 0.055, 0.1], np.float32) * (1 + 0.35 * strokes[..., None])
    fol = fol + underside[..., None] * np.array([0.05, 0.08, 0.16], np.float32)
    occl = cv2.GaussianBlur(soft, (0, 0), 3 * s + 1)
    fol = fol * (1 - 0.35 * occl[..., None])
    out = fol * (1 - soft[..., None]) + sc * soft[..., None]
    a2 = np.maximum(a, soft * (a > 0.01) + soft * (cv2.GaussianBlur((a > 0.5).astype(np.float32), (0, 0), 3 * s) > 0.2))
    a2 = np.clip(a2, 0, 1)
    # silhouette rims from the alpha gradient
    ab = cv2.GaussianBlur(a2, (0, 0), 1.6 * s + 0.4)
    gx = cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gn, -gy / gn            # outward normal
    edge = np.clip(gn * (3.0 * s + 0.8), 0, 1) * a2
    sky = np.clip(nx * sky_dir[0] + ny * sky_dir[1], 0, 1)
    rim_sky = edge * sky ** 1.5
    out = out + rim_sky[..., None] * np.array([0.22, 0.32, 0.62], np.float32)
    if lamp_uv is not None:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        lxv, lyv = lamp_uv[0] - xs, lamp_uv[1] - ys
        ln = np.sqrt(lxv * lxv + lyv * lyv) + 1e-3
        facing = np.clip((nx * lxv + ny * lyv) / ln, 0, 1)
        fall = np.exp(-ln / (W * 0.3))
        rim_w = edge * facing ** 3 * fall * (1 - 0.7 * soft)
        out = out + rim_w[..., None] * np.array([0.85, 0.52, 0.24], np.float32) * 0.55
    return (out * a2[..., None]).astype(np.float32), a2.astype(np.float32)


def wall_tex(w, h, planks, base, seed, snow_amb=(0.26, 0.36, 0.62)):
    """Clapboard wall texture (h, w, 3), linear-light reflectance*light already folded in:
    each plank has a lit lower lip (sky sheen) and a dark shadow line under the plank above, weathering
    streaks, a cool snow-bounce gradient near the ground and a dark band under the eave."""
    w, h = max(int(w), 2), max(int(h), 2)
    rng = np.random.default_rng(seed)
    base = np.asarray(base, np.float32)
    amb = np.asarray(snow_amb, np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    ph = np.mod(y * planks, 1.0)
    px_per = h / max(planks, 1)
    aa = min(1.0, 2.0 / max(px_per, 1e-3))                 # filter width (plank phase units)
    shadow = np.clip(1 - ph / max(0.14, aa), 0, 1)
    lip = np.clip((ph - 0.72) / max(0.12, aa), 0, 1) * np.clip((1.0 - ph) / max(0.08, aa), 0, 1)
    fade = np.clip(1.4 - aa * 2.5, 0, 1)                    # planks fade out when sub-pixel
    streak = cv2.GaussianBlur(rng.random((1, w)).astype(np.float32), (0, 0), max(w * 0.01, 0.5))
    streak = (streak - streak.mean()) / (streak.std() + 1e-6)
    board_v = rng.normal(0, 1, planks + 2)[np.clip((y * planks).astype(int), 0, planks + 1)]
    col = base * (1 + 0.06 * board_v[..., None] + 0.05 * streak[..., None])
    col = col * (1 - 0.45 * shadow * fade)[..., None] + (lip * fade)[..., None] * amb * 0.09
    bounce = np.clip((y - 0.55) / 0.45, 0, 1) ** 1.5
    col = col + bounce[..., None] * amb * 0.05
    eave = np.clip(1 - y / 0.22, 0, 1)
    col = col * (1 - 0.5 * eave[..., None])
    return np.broadcast_to(col, (h, w, 3)).astype(np.float32)


def vending_side(tw=60, th=140):
    """Side of the vending machine facing the camera: red body, white band, a lit advert panel with a
    big blue drink bottle and 'つめた～い' lettering. Reflectance-ish RGB (th, tw, 3); caller lights it."""
    from PIL import Image, ImageDraw
    S = 3
    W, H = tw * S, th * S
    im = Image.new('RGB', (W, H), (150, 14, 18))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(H * 0.13), W, int(H * 0.18)], fill=(235, 235, 235))
    # advert panel
    x0, y0, x1, y1 = int(W * 0.12), int(H * 0.22), int(W * 0.88), int(H * 0.62)
    d.rectangle([x0, y0, x1, y1], fill=(240, 244, 250))
    cx = (x0 + x1) // 2
    bw = int((x1 - x0) * 0.22)
    d.rounded_rectangle([cx - bw, int(H * 0.3), cx + bw, int(H * 0.56)], radius=bw // 2, fill=(40, 110, 210))
    d.rectangle([cx - bw // 2, int(H * 0.26), cx + bw // 2, int(H * 0.31)], fill=(230, 230, 235))
    d.rectangle([cx - bw, int(H * 0.39), cx + bw, int(H * 0.45)], fill=(250, 250, 250))
    d.text((cx, int(H * 0.245)), 'つめた～い', font=_font(H * 0.035), fill=(20, 80, 200), anchor='mm')
    d.rectangle([x0, int(H * 0.66), x1, int(H * 0.7)], fill=(240, 200, 60))
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
