"""
Inference pipeline:
  load model → predict Bézier curves → add noise → render → stitch → page → PDF.

Uses PIL/numpy throughout — cv2 is not required.
"""
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from bezier import label_to_curves, draw_bezier, NUM_CHARS
from model  import HandwritingCNN
from data   import CHAR_TO_IDX

_HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(_HERE, '..', 'checkpoints', 'generator.pt')

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
    model = HandwritingCNN(num_chars=ckpt['num_chars']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt, device


# ── curve prediction ──────────────────────────────────────────────────────────
def predict_curves(model, img_pil, char_str, device):
    """CNN forward pass: PIL image + char string → list of 5 Bézier curves."""
    tensor   = _transform(img_pil).unsqueeze(0).to(device)
    char_idx = torch.tensor([CHAR_TO_IDX[char_str]], device=device)
    with torch.no_grad():
        label = model(tensor, char_idx).squeeze().cpu().numpy()
    return label_to_curves(label)


def add_variation(curves, noise_scale=0.018):
    """
    Perturb only inner control points P1, P2.
    Endpoints P0 & P3 stay fixed so adjacent curves connect cleanly.
    noise_scale: 0.010=neat  0.018=natural  0.035=casual
    """
    return [
        [
            (x + (np.random.normal(0, noise_scale) if j in (1, 2) else 0),
             y + (np.random.normal(0, noise_scale) if j in (1, 2) else 0))
            for j, (x, y) in enumerate(curve)
        ]
        for curve in curves
    ]


def render_curves(curves, size=128, thickness=3):
    """Draw Bézier curves onto a white canvas, return uint8 numpy array."""
    canvas = np.ones((size, size), dtype=np.uint8) * 255
    return draw_bezier(canvas, curves, thickness)


# ── single character ──────────────────────────────────────────────────────────
def generate_character(model, char_str, ref_img_pil=None,
                        noise_scale=0.02, device=None):
    """
    Generate one handwritten character image (128×128 uint8 numpy array).
    If ref_img_pil is None, a blank canvas is used as reference.
    """
    if ref_img_pil is None:
        ref_img_pil = Image.fromarray(
            np.ones((128, 128), dtype=np.uint8) * 255, mode='L'
        )
    curves = predict_curves(model, ref_img_pil, char_str, device)
    varied = add_variation(curves, noise_scale)
    return render_curves(varied)


# ── word / page assembly ──────────────────────────────────────────────────────
def stitch_word(char_images, spacing=14, jitter_range=5):
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
    """Place word strip on a ruled page — pure PIL, no cv2 needed."""
    page = Image.fromarray(np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255, mode='L')
    draw = ImageDraw.Draw(page)

    # Ruled lines (light grey)
    for ry in range(160, canvas_h, 42):
        draw.line([(40, ry), (canvas_w - 40, ry)], fill=195, width=1)

    page_arr                   = np.array(page)
    ph, pw                     = strip.shape
    pw                         = min(pw, canvas_w - 65)
    page_arr[115:115 + ph, 65:65 + pw] = strip[:, :pw]
    return page_arr


def generate_word(model, word, noise_scale=0.02, device=None):
    """Generate each character of word and stitch into a page."""
    chars = [
        generate_character(model, ch, noise_scale=noise_scale, device=device)
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
    p.add_argument('text',  help='Text to generate (A-Z a-z 0-9 only)')
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--out',   default='outputs/generated')
    args = p.parse_args()

    model, ckpt, device = load_model()
    print(f"Model loaded (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")

    strip, page = generate_word(model, args.text, noise_scale=args.noise, device=device)
    if page is None:
        print('No supported characters.')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    Image.fromarray(page).save(args.out + '.png')
    export_pdf(page, args.out + '.pdf')
    print(f"Saved {args.out}.png  and  {args.out}.pdf")
