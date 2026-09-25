"""s04_rain_street real Japanese signage: word pools + clean anti-aliased text masks rendered with a real
Japanese font (Yu Gothic Bold) via Pillow. Replaces the old pseudo-kanji glyph generator.

All masks are rendered supersampled and area-downsampled, so small copy stays clean (no ragged edges) and
large sign faces keep crisp, even strokes. Vertical text is set one character per cell (tategaki), with the
long-vowel mark / wave dash rotated and small kana nudged to the upper right, as in real vertical signage."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = 'C:/Windows/Fonts'
_FONTS = {'bold': ['YuGothB.ttc', 'meiryob.ttc', 'msgothic.ttc'],
          'med': ['YuGothM.ttc', 'meiryo.ttc', 'msgothic.ttc']}
_font_cache = {}
_mask_cache = {}

# --------------------------------------------------------------------------------------------- vocabulary
# Only real, correctly written Japanese words/phrases that plausibly appear on a Tokyo side street.
POOLS = {
    'shop': {
        1: ['酒', '鮨', '麺', '茶', '串', '薬', '本', '宴'],
        2: ['薬局', '喫茶', '焼肉', '寿司', '酒場', '本屋', '旅館', '食堂', '珈琲', '焼鳥', '餃子', '定食', '中華',
            '蕎麦', '銭湯', '洋食', '質屋', '花屋', '雀荘', '酒処', '麺処', '天丼', '整体', '理容'],
        3: ['居酒屋', '喫茶店', '焼鳥屋', '定食屋', '美容室', '蕎麦屋', '串カツ', '焼肉屋', '古本屋', '寿司屋',
            '珈琲館', '洋食屋', 'おでん', 'くすり', 'ホテル', '整骨院', '餃子屋'],
        4: ['ラーメン', 'カラオケ', '中華料理', '大衆酒場', '立ち飲み', 'スナック', '回転寿司', '生ビール',
            'やきとり', 'お好み焼', '焼肉酒場', '珈琲専門', 'ドラッグ', '炉端焼き', '中華そば'],
        5: ['自動販売機', 'お食事処', '深夜営業中', 'ラーメン屋', '焼き鳥酒場', 'カラオケ館', '大衆居酒屋',
            '手打ち蕎麦', '本日営業中', '二十四時間'],
        6: ['炭火やきとり', '焼肉ホルモン', 'カラオケ本舗', 'ラーメン横丁', '串焼き居酒屋', '生ビール半額'],
        7: ['ゲームセンター', 'ドラッグストア', '立ち飲み居酒屋', '二十四時間営業', 'ご宴会承ります', 'テイクアウト可'],
        8: ['カラオケボックス', '毎日元気に営業中', '深夜二時まで営業'],
        9: ['生ビール一杯三百円', '昭和四十年創業の味', '焼き鳥とおでんの店'],
        10: ['ランチ営業・全席禁煙', '冷たいビールあります'],
        11: ['ランチ営業・禁煙席あり'],
        12: ['営業時間十七時〜二十四時'],
    },
    'addr': {2: ['本町', '栄町', '中町'], 3: ['一丁目', '二丁目', '三丁目', '四丁目', '五丁目']},
    'lantern': {1: ['酒', '祭', '串', '宴']},
    'noren': {1: ['酒', '鮨', '麺', '串', '呑']},
    'vend_head': {4: ['つめたい'], 5: ['つめた〜い', '自動販売機'], 6: ['あったか〜い', '冷たい飲み物']},
    'vend_panel': {3: ['千円札'], 4: ['お茶と水', '千円札可'], 5: ['電子マネー', 'つめた〜い']},
    'vend_can': {2: ['緑茶', '麦茶', 'お茶'], 3: ['コーラ', '天然水', '炭酸水'], 4: ['サイダー', 'ほうじ茶']},
    'vend_v': {3: ['天然水', '冷たい'], 4: ['つめたい', 'おいしい'], 5: ['冷えてます'], 6: ['つめたいお茶']},
}
# re-bucket every pool strictly by character count (robust to hand-editing mistakes)
for _name, _p in POOLS.items():
    _b = {}
    for _ws in _p.values():
        for _w in _ws:
            _b.setdefault(len(_w), []).append(_w)
    POOLS[_name] = _b

ROAD = '止まれ'

_ROTATE_V = set('ー〜～…‥―－-')
_SMALL = set('ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ')


def pick(rng, count, pool='shop'):
    """A real word/phrase of at most `count` characters (the longest lengths are preferred)."""
    P = POOLS.get(pool) or POOLS['shop']
    count = max(1, int(count))
    lens = sorted(k for k in P if k <= count)
    if not lens:
        lens = [min(P)]
    L = lens[-1]
    if len(lens) > 1 and rng.random() < 0.3:
        L = lens[-2]
    ws = P[L]
    return ws[int(rng.integers(len(ws)))]


def _font(px, kind='bold'):
    key = (int(px), kind)
    f = _font_cache.get(key)
    if f is None:
        for name in _FONTS[kind]:
            p = os.path.join(_FONT_DIR, name)
            if os.path.exists(p):
                f = ImageFont.truetype(p, int(px), index=0)
                break
        _font_cache[key] = f
    return f


def _char_img(ch, cell, vertical, kind):
    """Render one character centred in a square cell (supersampled 'L' image of size cell)."""
    im = Image.new('L', (cell, cell), 0)
    d = ImageDraw.Draw(im)
    f = _font(cell * 0.92, kind)
    if vertical and ch in _ROTATE_V:
        d.text((cell / 2, cell / 2), ch, fill=255, font=f, anchor='mm')
        im = im.transpose(Image.Transpose.ROTATE_270)
        return np.asarray(im)
    d.text((cell / 2, cell / 2), ch, fill=255, font=f, anchor='mm')
    a = np.asarray(im)
    if vertical and ch in _SMALL:
        s = int(cell * 0.14)
        a = np.roll(np.roll(a, -s, 0), s, 1)
    return a


def text_line(text, cell, vertical=False, kind='bold'):
    """Mask (float 0..1) of `text` set in square cells of `cell` px: (cell, n*cell) or (n*cell, cell)."""
    cell = max(2, int(cell))
    key = (text, cell, vertical, kind)
    m = _mask_cache.get(key)
    if m is not None:
        return m
    ss = 4 if cell < 24 else (2 if cell < 96 else 1)
    c = cell * ss
    ims = [_char_img(ch, c, vertical, kind) for ch in text]
    big = np.vstack(ims) if vertical else np.hstack(ims)
    big = big.astype(np.float32) / 255.0
    n = len(text)
    out = cv2.resize(big, ((cell if vertical else cell * n), (cell * n if vertical else cell)),
                     interpolation=cv2.INTER_AREA)
    # sharpen the coverage very slightly so small text stays legible rather than grey
    out = np.clip((out - 0.5) * 1.15 + 0.5, 0, 1) if cell >= 10 else np.clip(out * 1.2, 0, 1)
    _mask_cache[key] = out
    return out


def column(rng, w, h, count, margin=0.12, text=None, pool='shop', kind='bold'):
    """Vertical run of real text filling a (h, w) mask (drop-in for the old glyph_column)."""
    m = np.zeros((h, w), np.float32)
    g = int(w * (1 - 2 * margin))
    if g < 3 or count < 1:
        return m
    if text is None:
        text = pick(rng, count, pool)
    n = len(text)
    step = (h - 2 * margin * w) / max(count, n)
    g = int(min(g, step * 0.94))
    if g < 2:
        return m
    step = min(step, g * 1.12)          # tight tracking; the run is centred
    tm = text_line(text, g, vertical=True, kind=kind)
    # characters sit on the cell pitch `step`; the run is centred in the column
    y00 = margin * w + (h - 2 * margin * w - n * step) / 2
    x = int((w - g) / 2)
    for i in range(n):
        y = int(round(y00 + i * step + (step - g) / 2))
        gl = tm[i * g:(i + 1) * g]
        yy0, yy1 = max(y, 0), min(y + g, h)
        if yy1 <= yy0:
            continue
        m[yy0:yy1, x:x + g] = np.maximum(m[yy0:yy1, x:x + g], gl[yy0 - y:yy1 - y, :w - x])
    return m


def row(rng, w, h, count, margin=0.12, text=None, pool='shop', kind='bold'):
    """Horizontal run of real text filling a (h, w) mask (drop-in for the old glyph row)."""
    m = np.zeros((h, w), np.float32)
    g = int(h * (1 - 2 * margin))
    if g < 3 or count < 1:
        return m
    if text is None:
        # long fascia copy keeps ~10 % slack at each end so it never butts against a wall edge
        text = pick(rng, count if count < 5 else max(4, int(count * 0.8)), pool)
    n = len(text)
    step = (w - 2 * margin * h) / max(count, n)
    g = int(min(g, step * 0.96))
    if g < 2:
        return m
    step = min(step, g * 1.08)
    tm = text_line(text, g, vertical=False, kind=kind)
    x00 = margin * h + (w - 2 * margin * h - n * step) / 2
    y = int((h - g) / 2)
    for i in range(n):
        x = int(round(x00 + i * step + (step - g) / 2))
        gl = tm[:, i * g:(i + 1) * g]
        xx0, xx1 = max(x, 0), min(x + g, w)
        if xx1 <= xx0:
            continue
        m[y:y + g, xx0:xx1] = np.maximum(m[y:y + g, xx0:xx1], gl[:h - y, xx0 - x:xx1 - x])
    return m


def single(rng, n, pool='noren'):
    """One real character centred in an n x n mask."""
    P = POOLS[pool][1]
    ch = P[int(rng.integers(len(P)))]
    return text_line(ch, n, vertical=False).copy()
