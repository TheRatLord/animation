"""Foreground helpers for s03_city_dusk: real Japanese signage rendered with a system font (Yu Gothic),
returned as anti-aliased coverage masks to be painted into the supersampled near-plane canvas."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONTS = [r'C:\Windows\Fonts\YuGothB.ttc', r'C:\Windows\Fonts\YuGothM.ttc', r'C:\Windows\Fonts\msgothic.ttc',
          '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']


def _font(px):
    for p in _FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, int(px))
            except OSError:
                continue
    return ImageFont.load_default()


def text_mask(text, px, vertical=False, spacing=0.08, stroke=0):
    """Coverage mask (h, w) float32 0..1 of `text` with glyph size ~px (em height in pixels).
    vertical=True stacks the characters top to bottom (tategaki)."""
    f = _font(px)
    if not vertical:
        bb = f.getbbox(text, stroke_width=stroke)
        w, h = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
        im = Image.new('L', (max(w, 1), max(h, 1)), 0)
        ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), text, font=f, fill=255, stroke_width=stroke,
                                stroke_fill=255)
        return np.asarray(im, np.float32) / 255.0
    cells = []
    for ch in text:
        bb = f.getbbox(ch, stroke_width=stroke)
        cw = int(px * 1.05)
        im = Image.new('L', (cw, int(px * (1 + spacing))), 0)
        x = (cw - (bb[2] - bb[0])) // 2 - bb[0]
        y = (int(px * (1 + spacing)) - (bb[3] - bb[1])) // 2 - bb[1]
        ImageDraw.Draw(im).text((x, y), ch, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        cells.append(np.asarray(im, np.float32) / 255.0)
    return np.concatenate(cells, 0)


def fit(mask, w, h):
    """resize mask to exactly (h, w) keeping it crisp (area filter)"""
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    im = im.resize((max(int(w), 1), max(int(h), 1)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0
