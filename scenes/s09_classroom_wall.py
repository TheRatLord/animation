"""Front-wall decals for s09_classroom: cork notice board with pinned printouts (timetable etc.),
class-goal banners above the blackboard. Painted procedurally with Pillow (real Japanese text).

Returns an (h, w, 4) float32 array over front-wall coords (x = z * ppm, y = (HC - y) * ppm):
premultiplied RGB + alpha.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FB = 'C:/Windows/Fonts/YuGothB.ttc'
FM = 'C:/Windows/Fonts/YuGothM.ttc'
if not os.path.exists(FM):
    FM = 'C:/Windows/Fonts/meiryo.ttc'
if not os.path.exists(FB):
    FB = FM


def _font(path, px):
    return ImageFont.truetype(path, max(6, int(px)))


def _sheet(wm, hm, ppm, bg, draw_fn):
    """a sheet of paper wm x hm metres (RGBA) drawn by draw_fn(dr, W, H, s) with s = px per metre."""
    ss = 3
    s = ppm * ss
    W, H = int(wm * s), int(hm * s)
    im = Image.new('RGBA', (W, H), bg + (255,))
    dr = ImageDraw.Draw(im)
    draw_fn(dr, W, H, s)
    return im.resize((max(1, W // ss), max(1, H // ss)), Image.LANCZOS)


def _text_lines(dr, x, y, w, n, s, lh, rng, col=(95, 95, 100)):
    """lines of small body text (real Japanese, too small to matter but printed shapes read right)"""
    body = ['今週の予定について', '体育祭の練習が始まります', '提出物は金曜日までに', '朝読書の時間を大切に',
            '保護者会のお知らせ', '期末テストの範囲', '持ち物の確認をしよう', '掃除当番は交代制です']
    f = _font(FM, lh * 0.78 * s)
    for i in range(n):
        t = body[int(rng.integers(len(body)))]
        dr.text((x, y + i * lh * s), t, fill=col, font=f)


def _timetable(dr, W, H, s):
    fb = _font(FB, 0.045 * s)
    dr.text((W * 0.5, 0.012 * s), '時間割', fill=(40, 40, 60), font=fb, anchor='ma')
    dr.text((W * 0.93, 0.03 * s), '3年2組', fill=(70, 70, 90), font=_font(FM, 0.018 * s), anchor='ra')
    days = ['', '月', '火', '水', '木', '金']
    subj = [['国語', '数学', '英語', '理科', '社会'], ['数学', '英語', '体育', '国語', '数学'],
            ['英語', '理科', '数学', '社会', '英語'], ['社会', '国語', '音楽', '数学', '理科'],
            ['理科', '体育', '美術', '英語', '国語'], ['学活', '総合', '技術', '道徳', '体育']]
    x0, x1 = 0.02 * s, W - 0.02 * s
    y0, y1 = 0.075 * s, H - 0.02 * s
    nc, nr = 6, 7
    cw = (x1 - x0) / nc
    rh = (y1 - y0) / nr
    colors = {'国語': (255, 214, 214), '数学': (210, 226, 255), '英語': (255, 240, 200), '理科': (214, 245, 214),
              '社会': (240, 222, 255), '体育': (255, 228, 205)}
    fs = _font(FM, rh * 0.5)
    for r in range(nr):
        for c in range(nc):
            xa, ya = x0 + c * cw, y0 + r * rh
            fill = None
            if r == 0 and c > 0:
                fill = (70, 90, 150)
            elif r > 0 and c > 0:
                fill = colors.get(subj[r - 1][c - 1], (250, 250, 248))
            elif c == 0 and r > 0:
                fill = (225, 225, 230)
            if fill:
                dr.rectangle([xa, ya, xa + cw, ya + rh], fill=fill)
            if r == 0 and c > 0:
                dr.text((xa + cw / 2, ya + rh / 2), days[c], fill=(255, 255, 255), font=fs, anchor='mm')
            elif c == 0 and r > 0:
                dr.text((xa + cw / 2, ya + rh / 2), str(r), fill=(60, 60, 70), font=fs, anchor='mm')
            elif r > 0:
                dr.text((xa + cw / 2, ya + rh / 2), subj[r - 1][c - 1], fill=(50, 50, 60), font=fs, anchor='mm')
    lw = max(1, int(0.0018 * s))
    for r in range(nr + 1):
        dr.line([x0, y0 + r * rh, x1, y0 + r * rh], fill=(80, 80, 90), width=lw)
    for c in range(nc + 1):
        dr.line([x0 + c * cw, y0, x0 + c * cw, y1], fill=(80, 80, 90), width=lw)


def _news(title, sub, accent, seed):
    rng = np.random.default_rng(seed)

    def fn(dr, W, H, s):
        dr.rectangle([0, 0, W, 0.07 * s], fill=accent)
        dr.text((W * 0.5, 0.035 * s), title, fill=(255, 255, 255), font=_font(FB, 0.042 * s), anchor='mm')
        dr.text((0.02 * s, 0.085 * s), sub, fill=(60, 60, 70), font=_font(FB, 0.022 * s))
        _text_lines(dr, 0.02 * s, 0.125 * s, W, int((H / s - 0.15) / 0.024), s, 0.024, rng)
        # a little clip-art blob (a sun / flower) in a corner
        cx, cy, rr = W - 0.05 * s, H - 0.06 * s, 0.03 * s
        dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 190, 90))
        dr.ellipse([cx - rr * 0.5, cy - rr * 0.5, cx + rr * 0.5, cy + rr * 0.5], fill=(255, 230, 150))
    return fn


def _duty(dr, W, H, s):
    dr.text((W * 0.5, 0.012 * s), '給食当番表', fill=(160, 70, 40), font=_font(FB, 0.036 * s), anchor='ma')
    # a paper wheel chart: the classic rotating duty wheel
    cx, cy, R = W * 0.5, H * 0.58, min(W, H) * 0.36
    cols = [(255, 200, 200), (255, 236, 180), (200, 236, 200), (200, 220, 255), (236, 210, 250), (255, 220, 190)]
    for i in range(6):
        dr.pieslice([cx - R, cy - R, cx + R, cy + R], i * 60, i * 60 + 60, fill=cols[i], outline=(120, 110, 100),
                    width=max(1, int(0.002 * s)))
    names = ['配膳', '牛乳', 'おかず', 'ご飯', '食器', '片付け']
    f = _font(FM, 0.017 * s)
    for i, n in enumerate(names):
        a = math.radians(i * 60 + 30)
        dr.text((cx + math.cos(a) * R * 0.62, cy + math.sin(a) * R * 0.62), n, fill=(60, 50, 50), font=f,
                anchor='mm')
    dr.ellipse([cx - R * 0.12, cy - R * 0.12, cx + R * 0.12, cy + R * 0.12], fill=(240, 240, 235),
               outline=(120, 110, 100))


def _goal(big, head, border):
    def fn(dr, W, H, s):
        lw = max(2, int(0.01 * s))
        dr.rectangle([lw, lw, W - lw, H - lw], outline=border, width=lw)
        dr.text((0.04 * s, 0.03 * s), head, fill=border, font=_font(FB, 0.045 * s))
        n = len(big)
        fs = min(0.21 * s, (W - 0.1 * s) / (n + 0.6))
        dr.text((W * 0.5 + 0.04 * s, H * 0.58), big, fill=(30, 30, 35), font=_font(FB, fs), anchor='mm')
    return fn


def _paste(canvas, sheet, cx, cy, ang, rng, pins=True, shadow=0.45):
    """paste a sheet (RGBA) centred at canvas px (cx, cy), rotated ang degrees, with a soft drop shadow and
    push-pins at the top corners."""
    sh = sheet.rotate(ang, resample=Image.BICUBIC, expand=True)
    w, h = sh.size
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    a = np.array(sh)[..., 3].astype(np.float32) / 255.0
    sd = Image.fromarray((a * 255 * shadow).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(1, w * 0.012)))
    shadow_img = Image.new('RGBA', (w, h), (40, 30, 25, 0))
    shadow_img.putalpha(sd)
    off = max(1, int(w * 0.012))
    canvas.alpha_composite(shadow_img, (x0 + off, y0 + off))
    canvas.alpha_composite(sh, (x0, y0))
    if pins:
        dr = ImageDraw.Draw(canvas)
        sw, shh = sheet.size
        ca, sa = math.cos(math.radians(-ang)), math.sin(math.radians(-ang))
        pr = max(2, sw * 0.03)
        for fx in (-0.42, 0.42):
            lx, ly = fx * sw, -0.44 * shh
            px, py = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            col = [(220, 40, 50), (40, 110, 220), (240, 200, 30), (40, 170, 90)][int(rng.integers(4))]
            dr.ellipse([px - pr + pr * 0.5, py - pr + pr * 0.6, px + pr + pr * 0.5, py + pr + pr * 0.6],
                       fill=(40, 30, 25, 110))
            dr.ellipse([px - pr, py - pr, px + pr, py + pr], fill=col + (255,))
            dr.ellipse([px - pr * 0.45, py - pr * 0.6, px + pr * 0.05, py - pr * 0.1], fill=(255, 255, 255, 170))


def wall_decals(ppm, LZ, HC, seed=8):
    rng = np.random.default_rng(seed)
    w, h = int(LZ * ppm), int(HC * ppm)
    cv = Image.new('RGBA', (w, h), (0, 0, 0, 0))

    def P(z, y):
        return z * ppm, (HC - y) * ppm

    # --- cork notice board between the window corner and the blackboard
    z0, z1, y0, y1 = 0.24, 1.1, 1.0, 2.24
    a0, a1 = P(z0, y1), P(z1, y0)
    cw, chh = int(a1[0] - a0[0]), int(a1[1] - a0[1])
    cork = np.zeros((chh, cw, 4), np.uint8)
    n = rng.random((chh, cw))
    base = np.array([186, 136, 88], np.float32)
    tex = base[None, None, :] * (0.86 + 0.24 * n[..., None]) * (0.93 + 0.1 * rng.random((chh, 1, 1)))
    cork[..., :3] = np.clip(tex, 0, 255).astype(np.uint8)
    cork[..., 3] = 255
    fr = max(2, int(0.018 * ppm))
    cork[:fr, :, :3] = cork[-fr:, :, :3] = (196, 198, 202)
    cork[:, :fr, :3] = cork[:, -fr:, :3] = (196, 198, 202)
    cork[fr - 1:fr, fr:-fr, :3] = (150, 150, 155)
    corkim = Image.fromarray(cork, 'RGBA')
    # drop shadow of the board
    shp = Image.new('RGBA', (cw + 20, chh + 20), (0, 0, 0, 0))
    ImageDraw.Draw(shp).rectangle([10, 10, cw + 10, chh + 10], fill=(50, 40, 35, 90))
    shp = shp.filter(ImageFilter.GaussianBlur(4))
    cv.alpha_composite(shp, (int(a0[0]) - 6, int(a0[1]) - 6))
    cv.alpha_composite(corkim, (int(a0[0]), int(a0[1])))

    def put(sheet, zc, yc, ang, pins=True):
        x, y = P(zc, yc)
        _paste(cv, sheet, x, y, ang, rng, pins)

    put(_sheet(0.4, 0.5, ppm, (252, 252, 248), _timetable), 0.49, 1.93, rng.normal(0, 0.8))
    put(_sheet(0.3, 0.44, ppm, (250, 248, 240), _news('学級通信', '3年2組 第12号', (60, 120, 190), 3)),
        0.9, 1.96, rng.normal(0, 1.5))
    put(_sheet(0.34, 0.4, ppm, (253, 250, 244), _duty), 0.44, 1.32, rng.normal(0, 1.2))
    put(_sheet(0.28, 0.4, ppm, (255, 236, 238), _news('保健だより', '9月号 熱中症に注意', (220, 110, 130), 5)),
        0.87, 1.36, rng.normal(0, 1.5))
    put(_sheet(0.14, 0.1, ppm, (255, 250, 170), lambda dr, W, H, s: dr.text(
        (W * 0.5, H * 0.5), '提出！', fill=(200, 40, 40), font=_font(FB, 0.035 * s), anchor='mm')),
        1.0, 1.1, rng.normal(0, 4))
    # --- class goal banners above the blackboard (the clock sits between them)
    put(_sheet(1.45, 0.34, ppm, (253, 251, 245), _goal('一致団結', '学級目標', (200, 60, 50))),
        2.45, 2.45, rng.normal(0, 0.4), pins=False)
    put(_sheet(1.35, 0.3, ppm, (250, 252, 245), _goal('時間を守ろう', '今月の目標', (40, 110, 70))),
        4.62, 2.44, rng.normal(0, 0.4), pins=False)
    arr = np.array(cv).astype(np.float32) / 255.0
    a = arr[..., 3:4]
    rgb = arr[..., :3] * a                    # albedo, premultiplied
    return np.concatenate([rgb, a], -1).astype(np.float32)
