"""
Inference pipeline:
  load model → render font char → CNN translates to handwriting style
  → stitch word → build ruled page → optional PDF export
"""
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from model       import HandwritingStyleNet
from data        import CHAR_TO_IDX, IDX_TO_CHAR
from font_render import render_char

_HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(_HERE, '..', 'checkpoints', 'style_net.pt')

_transform = T.Compose([
    T.Grayscale(1),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = ckpt_path or CKPT_PATH
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HandwritingStyleNet(num_writers=ckpt['num_writers']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt, device


# ── single character generation ───────────────────────────────────────────────
def generate_char(model, char, writer_idx, device):
    """
    Render char with font → CNN translates to writer's handwriting style.
    Returns uint8 numpy array (128, 128).
    """
    font_img = render_char(char)
    tensor   = _transform(font_img).unsqueeze(0).to(device)
    widx     = torch.tensor([writer_idx], device=device)

    with torch.no_grad():
        out = model(tensor, widx)   # (1, 1, 128, 128), range [-1, 1]

    arr = ((out.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return arr


# ── word / page assembly ──────────────────────────────────────────────────────
def stitch_word(char_images, spacing=10, jitter_range=5):
    H       = 128
    total_w = sum(img.shape[1] for img in char_images) + spacing * (len(char_images) - 1)
    strip   = np.ones((H + 30, total_w), dtype=np.uint8) * 255
    x = 0
    for ch in char_images:
        jitter = random.randint(-jitter_range, jitter_range)
        y      = 10 + jitter
        strip[y:y + H, x:x + ch.shape[1]] = ch
        x += ch.shape[1] + spacing
    return strip


def build_page(strip, canvas_h=420, canvas_w=950):
    page = Image.fromarray(np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255, mode='L')
    draw = ImageDraw.Draw(page)
    for ry in range(160, canvas_h, 42):
        draw.line([(40, ry), (canvas_w - 40, ry)], fill=195, width=1)
    page_arr            = np.array(page)
    ph, pw              = strip.shape
    pw                  = min(pw, canvas_w - 65)
    page_arr[115:115 + ph, 65:65 + pw] = strip[:, :pw]
    return page_arr


def generate_word(model, word, writer_idx=0, device=None):
    """Generate all characters of word and stitch onto a ruled page."""
    chars = [
        generate_char(model, ch, writer_idx, device)
        for ch in word if ch in CHAR_TO_IDX
    ]
    if not chars:
        return None, None
    strip = stitch_word(chars)
    page  = build_page(strip)
    return strip, page


# ── PDF export ────────────────────────────────────────────────────────────────
def export_pdf(page_np, pdf_path):
    from reportlab.pdfgen import canvas as rl_canvas
    tmp = pdf_path.replace('.pdf', '_tmp.png')
    Image.fromarray(page_np).save(tmp)
    iw, ih = page_np.shape[1], page_np.shape[0]
    c = rl_canvas.Canvas(pdf_path, pagesize=(iw, ih))
    c.drawImage(tmp, 0, 0, iw, ih)
    c.save()
    os.remove(tmp)


# ── quick CLI demo ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('text',         help='Text to generate (A-Z a-z 0-9)')
    p.add_argument('--writer',     type=int, default=0, help='Writer index (0-5)')
    p.add_argument('--out',        default='outputs/generated')
    args = p.parse_args()

    model, ckpt, device = load_model()
    print(f"Loaded: epoch={ckpt['epoch']}, val_L1={ckpt['val_loss']:.4f}")
    print(f"Writers: {ckpt['writer_names']}")

    strip, page = generate_word(model, args.text, writer_idx=args.writer, device=device)
    if page is None:
        print('No supported characters.')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    Image.fromarray(page).save(args.out + '.png')
    export_pdf(page, args.out + '.pdf')
    print(f"Saved {args.out}.png  and  {args.out}.pdf")
