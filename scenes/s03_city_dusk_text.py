"""Real Japanese text atlases for the s03_city_dusk city renderer.

The numba facade shader cannot call a font renderer, so the shop-sign and billboard lettering is pre-rendered
here with a real system font (Yu Gothic Bold via Pillow) into small coverage atlases with a pre-filtered
mip chain. The shader samples them trilinearly by pixel footprint, so tiny far signs read as properly
anti-aliased real words (never pseudo-kanji blocks), and because the city plates are rendered once and only
translated, there is no temporal shimmer.

  VA  (L, NV, VC*R, R)  vertical (tategaki) shop-sign words, one char per R x R cell, top-aligned
  VN  (NV,)             char count per vertical word
  HA  (L, 2*NH, R, HW)  billboard lines: entry 2k = big headline, 2k+1 = small sub line (centred)
  HN  (2*NH,)           natural text width of each line in texels (for aspect-correct fitting)
  NA, NN                rooftop company-name boards, same layout as HA / HN
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw

import s03_city_dusk_fg as FGH

R = 24          # texels per character cell
VC = 8          # max chars per vertical sign
HW = 8 * R      # billboard line width in texels
L = 6           # mip levels

# common, correct shop / building signage (vertical)
VWORDS = ['居酒屋', '薬局', 'ラーメン', 'カラオケ', '歯科医院', '不動産', '喫茶店', 'ホテル', '焼肉', '寿司',
          '駐車場', '学習塾', '美容室', '眼科', '中華料理', '質屋', '整骨院', '英会話', '焼き鳥', '定食屋',
          'そば処', '珈琲', '古本', 'ビジネスホテル', '内科小児科', 'パチンコ', '麻雀', 'クリーニング',
          '銭湯', '書店', '花屋', 'ゲームセンター']

# rooftop billboards: (headline, sub line)
HPAIRS = [('夏の旅へ', 'ようこそ東京'), ('新発売', '冷たい緑茶'), ('空をみあげて', 'ひろがる未来'),
          ('花火大会', '八月十日 開催'), ('音楽祭', '今夜 開演'), ('おいしい水', '天然水'),
          ('未来へ', '東雲グループ'), ('お引越し', '無料見積り'), ('英会話', 'はじめよう'),
          ('ありがとう', '創業三十周年')]


# rooftop company-name boards (fictional, plausible names)
NAMES = ['東雲電機', 'さくら薬局', 'ホテル青葉', '中央信用金庫', '日の出生命', '大和建設', '光が丘病院',
         '朝日不動産', '松田歯科医院', '北斗証券', '青空保険', '若葉学院', '丸八商事', 'みどり銀行', '星野ビル',
         'ひまわり保育園', '新宿第一ビル', '東雲ホテル']


def _line(txt):
    m = FGH.text_mask(txt, R * 4)
    h, ww = m.shape
    s = (R * 0.8) / h
    if ww * s > HW - 4:
        s = (HW - 4) / ww
    m = cv2.resize(m, (max(int(ww * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
    h, ww = m.shape
    out = np.zeros((R, HW), np.float32)
    y0, x0 = (R - h) // 2, (HW - ww) // 2
    out[y0:y0 + h, x0:x0 + ww] = m
    return out, ww + 2


def _mips(img):
    """pre-filtered chain at the same size: level k ~ box filter of 2^k texels"""
    h, w = img.shape
    out = [img]
    for k in range(1, L):
        s = 2 ** k
        d = cv2.resize(img, (max(w // s, 1), max(h // s, 1)), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, 0)


def _cell(ch):
    """one character on a fixed em cell (R x R), centred by the font's own metrics so small kana stay
    small; the long-vowel mark is turned vertical for tategaki."""
    S = R * 4
    im = Image.new('L', (S, S), 0)
    f = FGH._font(int(S * 0.84))
    ImageDraw.Draw(im).text((S / 2, S / 2), ch, font=f, fill=255, anchor='mm')
    m = np.asarray(im, np.float32) / 255.0
    if ch in 'ー〜':
        m = np.ascontiguousarray(np.rot90(m, -1))
    return cv2.resize(m, (R, R), interpolation=cv2.INTER_AREA)


def build():
    # ---- vertical words: each char centred in its R x R cell
    VA = np.zeros((len(VWORDS), VC * R, R), np.float32)
    VN = np.zeros(len(VWORDS), np.int64)
    for i, w in enumerate(VWORDS):
        n = min(len(w), VC)
        VN[i] = n
        for j, ch in enumerate(w[:n]):
            VA[i, j * R:(j + 1) * R] = _cell(ch)
    # ---- billboard lines, centred, natural width recorded
    HA = np.zeros((2 * len(HPAIRS), R, HW), np.float32)
    HN = np.zeros(2 * len(HPAIRS), np.float64)
    for i, pair in enumerate(HPAIRS):
        for k, txt in enumerate(pair):
            HA[2 * i + k], HN[2 * i + k] = _line(txt)
    NA = np.zeros((len(NAMES), R, HW), np.float32)
    NN = np.zeros(len(NAMES), np.float64)
    for i, txt in enumerate(NAMES):
        NA[i], NN[i] = _line(txt)
    VAm = np.stack([_mips(v) for v in VA], 1).astype(np.float32)   # (L, NV, h, w)
    HAm = np.stack([_mips(v) for v in HA], 1).astype(np.float32)
    NAm = np.stack([_mips(v) for v in NA], 1).astype(np.float32)
    return (np.ascontiguousarray(VAm), VN, np.ascontiguousarray(HAm), HN, np.ascontiguousarray(NAm), NN)


_CACHE = None


def atlases():
    global _CACHE
    if _CACHE is None:
        _CACHE = build()
    return _CACHE
