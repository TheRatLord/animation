"""s05_sakura helpers: real Japanese lettering on the far-bank signboards.

Each signboard box (type 7) of the environment ray-cast gets a word rendered with a real Japanese font
(Pillow, Yu Gothic Bold / Meiryo / MS Gothic from C:/Windows/Fonts), mapped perspective-correctly through
the per-pixel world coordinates of its river-facing face (s, Y for faces facing the river, u, Y for faces
facing the camera).  Vertical signs (taller than wide) get tategaki (one character per row).  The letter
colour is derived from the already shaded + hazed board pixel, so the lettering sits in the same light
and aerial perspective as the board.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FONTS = ['C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/YuGothB.ttc', 'C:/Windows/Fonts/meiryob.ttc', 'C:/Windows/Fonts/meiryo.ttc',
         'C:/Windows/Fonts/msgothic.ttc']
WORDS_H = ['薬局', '喫茶', '花屋', '書店', '歯科', '銭湯', 'そば', '寿司', '眼科', '旅館', '珈琲', '酒店',
           'さくら', '花見', '和菓子', '居酒屋', '不動産', '美容室', 'ラーメン', 'パン屋', '桜まつり', '弁当']
# round 8: small boards only get simple, open glyphs (kana / few-stroke kanji) that stay legible at 15-25 px
SIMPLE_H = ['そば', 'さくら', 'パン', 'カフェ', 'うどん', 'すし', '本屋', 'くすり', 'たばこ']
SIMPLE_V = ['そば', 'すし', 'くすり', 'うどん', 'たばこ', 'パン']
WORDS_V = ['薬局', '喫茶', '歯科', '書店', '花屋', '銭湯', 'そば', '旅館', '眼科', '居酒屋', '不動産', '和菓子']


def _font(px):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, px, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_img(word, vertical, aspect, res=256):
    """Word rendered white on black filling a board of the given aspect (w/h) -> float mask (h, w)."""
    if vertical:
        h = res * 3
        w = max(8, int(h * aspect))
        n = len(word)
        px = int(min(w * 0.78, h * 0.86 / n))
        fnt = _font(max(8, px))
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        y0 = (h - px * n * 1.0) / 2
        for i, ch in enumerate(word):
            bb = d.textbbox((0, 0), ch, font=fnt)
            cw = bb[2] - bb[0]
            d.text(((w - cw) / 2 - bb[0], y0 + i * px - bb[1] * 0.5), ch, fill=255, font=fnt)
    else:
        h = res
        w = max(8, int(h * aspect))
        fnt = _font(int(h * 0.6))
        im = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(im)
        bb = d.textbbox((0, 0), word, font=fnt)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        sc = min(1.0, (w * 0.86) / max(tw, 1))
        if sc < 1.0:
            fnt = _font(max(8, int(h * 0.6 * sc)))
            bb = d.textbbox((0, 0), word, font=fnt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((w - tw) / 2 - bb[0], (h - th) / 2 - bb[1]), word, fill=255, font=fnt)
    return np.asarray(im, np.float32) / 255.0


def paint_signs(out, bid, face, uo, Yo, so, B, ss, min_px=60, seed=3, wscale=1.0):
    """In place on the supersampled env buffer out (Hs, Ws, 4).

    The word on each board depends only on the board's index (never on which boards happen to be visible /
    large enough at the current resolution), and the size thresholds are in 1080p pixels (wscale = W / 1920),
    so --stills, --strip and the half / full-res videos all letter the same boards with the same words."""
    rng = np.random.default_rng(seed)
    ids = np.nonzero(B[:, 6] == 7)[0]
    if not len(ids):
        return
    Hs, Ws = bid.shape
    hv = 0
    vv = 0
    h0 = int(rng.integers(0, len(WORDS_H)))
    v0 = int(rng.integers(0, len(WORDS_V)))
    for i in ids:
        u0, u1, Y0, Y1, s0, s1 = B[i, :6]
        bx0, bx1 = int(max(0, B[i, 12])), int(min(Ws, B[i, 13] + 1))
        by0, by1 = int(max(0, B[i, 14])), int(min(Hs, B[i, 15] + 1))
        if bx1 - bx0 < 3 or by1 - by0 < 3:
            continue
        m = bid[by0:by1, bx0:bx1] == i
        if m.sum() < min_px * ss * ss * wscale * wscale:
            continue
        fl_ = face[by0:by1, bx0:bx1]
        fa = np.abs(fl_[m])
        # the face to letter: the big one facing the river (|fc| 1) or the camera (|fc| 3)
        n1, n3 = int((fa == 1).sum()), int((fa == 3).sum())
        axis = 1 if n1 >= n3 else 3
        mm = m & (np.abs(fl_) == axis)
        if mm.sum() < min_px * ss * ss * wscale * wscale * 0.6:
            continue
        fsign = int(fl_[mm][0])
        ys, xs = np.nonzero(mm)
        ly0, ly1, lx0, lx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # a board partly hidden behind another board / building would show cut glyphs -> leave it plain
        full = max((B[i, 13] - B[i, 12] - 4) * (B[i, 15] - B[i, 14] - 4), 1.0)
        if (ly1 - ly0) * (lx1 - lx0) < 0.7 * full:
            continue
        mm = mm[ly0:ly1, lx0:lx1]
        y0, y1, x0, x1 = ly0 + by0, ly1 + by0, lx0 + bx0, lx1 + bx0
        if axis == 1:
            a_, b_ = s0, s1
            tc = so[y0:y1, x0:x1]
        else:
            a_, b_ = u0, u1
            tc = uo[y0:y1, x0:x1]
        width = b_ - a_
        height = Y1 - Y0
        vertical = height > width * 1.3
        if vertical:
            word = WORDS_V[(int(i) * 7 + v0) % len(WORDS_V)]
        else:
            nmax = int(np.clip(round(width / max(height, 1e-3) * 0.85), 2, 4))
            cands = [w_ for w_ in WORDS_H if len(w_) <= nmax and len(w_) >= min(nmax, 3) - 1]
            word = cands[(int(i) * 5 + h0) % len(cands)]
        # too small to letter legibly -> leave a plain panel (never pseudo-glyphs)
        glyph_px = ((x1 - x0) if vertical else (y1 - y0)) / ss / wscale * (0.78 if vertical else 0.6)
        if glyph_px < 14.0:
            continue
        if glyph_px < 26.0:
            if vertical:
                word = SIMPLE_V[(int(i) * 7 + v0) % len(SIMPLE_V)]
            else:
                nmax = int(np.clip(round(width / max(height, 1e-3) * 0.85), 2, 3))
                cands = [w_ for w_ in SIMPLE_H if len(w_) <= nmax]
                word = cands[(int(i) * 5 + h0) % len(cands)]
        tex = _text_img(word, vertical, width / max(height, 1e-3))
        th, tw = tex.shape
        # board margins (frame) -> text lives in the inner 90%
        tx = (tc - a_) / width
        if axis == 3:
            tx = 1 - tx if fsign > 0 else tx
        else:
            tx = tx if fsign > 0 else 1 - tx
        ty = (Y1 - Yo[y0:y1, x0:x1]) / height
        mapx = np.clip(tx * tw, 0, tw - 1).astype(np.float32)
        mapy = np.clip(ty * th, 0, th - 1).astype(np.float32)
        # prefilter the text to the local pixel footprint (sign seen small / at an angle)
        scale = max(tw / max(x1 - x0, 1), th / max(y1 - y0, 1))
        if scale > 1.5:
            tex = cv2.GaussianBlur(tex, (0, 0), 0.35 * scale)
        k = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR) * mm
        c = out[y0:y1, x0:x1, :3]
        lum = c.mean(-1, keepdims=True)
        light_board = lum > 0.8
        tcol = np.where(light_board, c * np.array([0.55, 0.32, 0.36], np.float32),
                        1.0 - (1.0 - c) * 0.12)
        c += (tcol - c) * k[..., None] * 0.95
