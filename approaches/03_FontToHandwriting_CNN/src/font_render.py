"""
Renders a single character as a clean 128×128 greyscale image
using a system TTF font — this is the "content" input to the CNN.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 128

_CANDIDATES = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_FONT_PATH = next((f for f in _CANDIDATES if os.path.exists(f)), None)


def _make_font(size_px):
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, size_px)
        except Exception:
            pass
    return ImageFont.load_default()


def render_char(char, canvas_size=SIZE, fill_ratio=0.60):
    """
    Returns a PIL Image (mode 'L', canvas_size×canvas_size):
    white background, black ink, character centred.
    """
    img  = Image.new('L', (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)

    font_px = max(10, int(canvas_size * fill_ratio))
    font    = _make_font(font_px)

    # Iteratively shrink until the character fits within canvas
    while font_px > 10:
        bbox = draw.textbbox((0, 0), char, font=font)
        cw, ch = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if cw <= canvas_size * 0.90 and ch <= canvas_size * 0.90:
            break
        font_px -= 4
        font = _make_font(font_px)

    bbox = draw.textbbox((0, 0), char, font=font)
    cw   = bbox[2] - bbox[0]
    ch   = bbox[3] - bbox[1]
    x    = (canvas_size - cw) // 2 - bbox[0]
    y    = (canvas_size - ch) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)
    return img


def render_char_array(char, canvas_size=SIZE):
    """Returns uint8 numpy array (H, W)."""
    return np.array(render_char(char, canvas_size))
