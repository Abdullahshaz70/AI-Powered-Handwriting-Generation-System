"""
Generate handwriting PNG samples for a list of sentences.
Run from project root:
    python generate_samples.py
    python generate_samples.py --noise 0.025 --out outputs/samples
"""
import os
import sys
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from generate import load_model, generate_char
from data     import CHAR_TO_IDX, CANVAS_SIZE

# ── sentences to render ───────────────────────────────────────────────────────
SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "Hello World",
    "Aa Bb Cc Dd Ee Ff Gg Hh Ii Jj",
    "Kk Ll Mm Nn Oo Pp Qq Rr Ss Tt",
    "Uu Vv Ww Xx Yy Zz",
]

# ── layout constants ──────────────────────────────────────────────────────────
PAGE_W     = 1400
MARGIN     = 60
CHAR_GAP   = 8
WORD_GAP   = 40
LINE_H     = CANVAS_SIZE        # 128px tall per line
LINE_STEP  = LINE_H + 36        # gap between baselines
RULE_COLOR = 200                # light grey ruled lines
BG_COLOR   = 255


def _stitch_word(char_imgs):
    """Horizontally stitch character images for one word."""
    w_total = sum(c.shape[1] for c in char_imgs) + CHAR_GAP * (len(char_imgs) - 1)
    strip   = np.ones((LINE_H, w_total), dtype=np.uint8) * BG_COLOR
    x = 0
    for c in char_imgs:
        jitter = random.randint(-3, 3)
        y_off  = max(0, min(jitter, LINE_H - c.shape[0]))
        strip[y_off:y_off + c.shape[0], x:x + c.shape[1]] = c
        x += c.shape[1] + CHAR_GAP
    return strip


def generate_sentence_png(model, sentence, device, noise_scale=0.018):
    """
    Render sentence onto a ruled page with automatic word-wrap.
    Returns uint8 numpy array.
    """
    # Build word strips
    word_strips = []
    for word in sentence.split(' '):
        char_imgs = [
            generate_char(model, ch, device, noise_scale)
            for ch in word if ch in CHAR_TO_IDX
        ]
        if char_imgs:
            word_strips.append(_stitch_word(char_imgs))

    if not word_strips:
        return None

    # Word-wrap into lines
    avail_w = PAGE_W - 2 * MARGIN
    lines   = []
    row     = []
    row_w   = 0
    for strip in word_strips:
        sw = strip.shape[1]
        if row and row_w + WORD_GAP + sw > avail_w:
            lines.append(row)
            row   = [strip]
            row_w = sw
        else:
            row.append(strip)
            row_w = row_w + WORD_GAP + sw if row_w else sw
    if row:
        lines.append(row)

    # Build page canvas
    page_h = MARGIN + len(lines) * LINE_STEP + MARGIN
    page   = np.ones((page_h, PAGE_W), dtype=np.uint8) * BG_COLOR

    # Draw ruled lines and place words
    for li, line_words in enumerate(lines):
        y_top  = MARGIN + li * LINE_STEP
        y_rule = y_top + LINE_H + 4           # rule just below the text
        page[y_rule, MARGIN:PAGE_W - MARGIN] = RULE_COLOR

        x = MARGIN
        for strip in line_words:
            h, w = strip.shape
            y2   = min(y_top + h, page_h)
            x2   = min(x + w,    PAGE_W - MARGIN)
            page[y_top:y2, x:x2] = strip[:y2 - y_top, :x2 - x]
            x += w + WORD_GAP

    return page


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', type=float, default=0.018,
                    help='Pen variation noise (default 0.018, try 0.01-0.04)')
    ap.add_argument('--out',   default='outputs/samples',
                    help='Output directory (default outputs/samples)')
    ap.add_argument('--ckpt',  default=None,
                    help='Checkpoint path (default checkpoints/style_net.pt)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Loading model...')
    model, ckpt, device = load_model(args.ckpt)
    print(f'  epoch={ckpt["epoch"]}, val_MSE={ckpt["val_loss"]:.6f}')

    for i, sentence in enumerate(SENTENCES):
        print(f'Generating [{i+1}/{len(SENTENCES)}]: {sentence[:50]}')
        page = generate_sentence_png(model, sentence, device, noise_scale=args.noise)
        if page is None:
            print('  (no supported characters, skipped)')
            continue
        fname = f'{i+1:02d}_{sentence[:30].replace(" ","_")}.png'
        path  = os.path.join(args.out, fname)
        Image.fromarray(page).save(path)
        print(f'  Saved {path}')

    print(f'\nDone — {len(SENTENCES)} images in {args.out}/')


if __name__ == '__main__':
    main()
