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


# ---------------------------------------------------------------------------------------------------------------
# round 13: painted concrete for the near rooftop building (wwy_09): panel seams, mottled weathering, rain-stain
# streaks running down from sills and seams, a cool shade that deepens toward the street and a warm spill from
# the afterglow along the top and the sun-side end.
import cv2  # noqa: E402


def _up(n, h, w):
    return cv2.resize(n.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def weather_wall(col, y0, y1, x0, x1, fx, fy, seed, ss, H, seam_fx, sills_fy, top_fy, sun_fx, rim):
    """In place on col[y0:y1, x0:x1] (supersampled plate). fx(px) / fy(py) map plate px -> frame fractions.
    seam_fx: list of vertical panel seam x (frame fractions); sills_fy: y of sills / floor joints (stains start
    there); top_fy: parapet top; sun_fx: frame x of the sun (warm spill side); rim: warm rim colour."""
    h, w = y1 - y0, x1 - x0
    if h <= 2 or w <= 2:
        return
    rng = np.random.default_rng(seed)
    X = fx(np.arange(x0, x1, dtype=np.float32))[None, :]
    Y = fy(np.arange(y0, y1, dtype=np.float32))[:, None]
    # mottled weathering (two scales)
    m1 = _up(rng.random((max(h // 90, 3), max(w // 90, 3))), h, w)
    m2 = _up(rng.random((max(h // 25, 3), max(w // 25, 3))), h, w)
    mot = 0.78 + 0.3 * m1 + 0.14 * m2
    # rain stains: vertically stretched noise, strongest just under the sills, fading downward
    st = _up(rng.random((max(h // 160, 2), max(w // 7, 3))), h, w)
    st = np.clip((st - 0.45) / 0.4, 0, 1)
    under = np.zeros((h, w), np.float32)
    for sy in sills_fy:
        d = (Y - sy) / 0.03
        under = np.maximum(under, np.where(d > 0, np.exp(-d), 0.0).astype(np.float32) * np.ones_like(X))
    stain = st * (0.25 + 0.75 * under)
    val = mot * (1.0 - 0.45 * stain)
    # panel seams: thin dark line with a faint lit lower / left lip
    px = 1.0 / (1920.0 * ss) * 1920.0 / max(H * 16 / 9, 1.0)
    lw = 1.3 / (H * 16 / 9)
    for sx in seam_fx:
        d = np.abs(X - sx)
        val = val * (1.0 - 0.35 * np.clip(1.0 - d / lw, 0, 1))
        dl = X - (sx - 1.8 * lw)
        val = val * (1.0 + 0.1 * np.clip(1.0 - np.abs(dl) / lw, 0, 1))
    c = col[y0:y1, x0:x1]
    # cool shade deepening toward the street, warm spill from the afterglow on the top and the sun-side end
    down = np.clip((Y - top_fy) / 0.14, 0, 1)
    cool = np.array([0.82, 0.9, 1.08], np.float32)
    c *= (val * (1.0 - 0.28 * down))[..., None]
    c *= (1 - 0.35 * down[..., None]) + 0.35 * down[..., None] * cool
    spill = 0.1 * np.exp(-np.maximum(Y - top_fy, 0) / 0.012) + 0.06 * np.exp(-np.abs(X - sun_fx) / 0.07) * \
        np.exp(-np.maximum(Y - top_fy, 0) / 0.08)
    c += (spill * (0.7 + 0.3 * m1))[..., None] * np.asarray(rim, np.float32) * 0.5
    col[y0:y1, x0:x1] = c
