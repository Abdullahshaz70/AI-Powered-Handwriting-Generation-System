"""
Inference pipeline:
  load model → pick real writer reference image → CNN predicts Bézier curves
  → add variation → render → stitch word → build ruled page → optional PDF
"""
import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model  import HandwritingCNN
from bezier import label_to_curves, add_variation, draw_bezier
from data   import CHAR_TO_IDX, scale_to_canvas, CANVAS_SIZE

_HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints', 'style_net.pt'))
DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', 'Data', 'Writers_pngs'))

SKIP_DIRS = {'Writers_Zip', 'output_preview', '__pycache__'}


def _load_writer_refs(data_root):
    """Return {writer_idx: [PIL.Image, ...]} for each writer folder."""
    refs = {}
    dirs = sorted(
        [e for e in os.scandir(data_root) if e.is_dir() and e.name not in SKIP_DIRS],
        key=lambda e: e.name,
    )
    for wid, entry in enumerate(dirs):
        imgs = [
            Image.open(os.path.join(entry.path, f)).convert('L')
            for f in sorted(os.listdir(entry.path))
            if f.lower().endswith('.png')
        ]
        refs[wid] = imgs
    return refs


def _img_to_tensor(pil_img, device):
    arr = np.array(pil_img.resize((CANVAS_SIZE, CANVAS_SIZE)))
    arr = scale_to_canvas(np.where(arr < 128, 0, 255).astype(np.uint8), target_fill=0.80)
    t   = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
    return t.to(device)


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path  = ckpt_path or CKPT_PATH
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = HandwritingCNN(num_chars=62).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    refs  = _load_writer_refs(DATA_ROOT)
    return model, ckpt, device, refs


# ── single character generation ───────────────────────────────────────────────
def generate_char(model, char, writer_refs, device, noise_scale=0.018):
    """
    Pick a real reference image from writer_refs, predict Bézier curves, render.
    Returns uint8 numpy array (128, 128).
    """
    ref_img  = random.choice(writer_refs)
    tensor   = _img_to_tensor(ref_img, device)
    char_idx = torch.tensor([CHAR_TO_IDX[char]], device=device)

    with torch.no_grad():
        label = model(tensor, char_idx).squeeze().cpu().numpy()   # (24,)

    curves = label_to_curves(label)
    varied = add_variation(curves, noise_scale=noise_scale)
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
    return draw_bezier(canvas, varied, thickness=3)


# ── word / page assembly ──────────────────────────────────────────────────────
def stitch_word(char_images, spacing=10, jitter_range=5):
    H       = CANVAS_SIZE
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
    page_arr = np.array(page)
    ph, pw   = strip.shape
    pw       = min(pw, canvas_w - 65)
    page_arr[115:115 + ph, 65:65 + pw] = strip[:, :pw]
    return page_arr


def generate_word(model, word, writer_idx=0, device=None, refs=None, noise_scale=0.018):
    """Generate all characters of word and stitch onto a ruled page."""
    writer_refs = refs.get(writer_idx, []) if refs else []
    if not writer_refs:
        raise ValueError(f'No reference images for writer {writer_idx}')

    chars = [
        generate_char(model, ch, writer_refs, device, noise_scale)
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
    p.add_argument('text',     help='Text to generate (A-Z a-z 0-9)')
    p.add_argument('--writer', type=int,   default=0,     help='Writer index (0-5)')
    p.add_argument('--noise',  type=float, default=0.018, help='Variation noise scale')
    p.add_argument('--out',    default='outputs/generated')
    args = p.parse_args()

    model, ckpt, device, refs = load_model()
    print(f"Loaded: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"Writers: {ckpt['writer_names']}")

    strip, page = generate_word(model, args.text, writer_idx=args.writer,
                                device=device, refs=refs, noise_scale=args.noise)
    if page is None:
        print('No supported characters.')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    Image.fromarray(page).save(args.out + '.png')
    export_pdf(page, args.out + '.pdf')
    print(f"Saved {args.out}.png  and  {args.out}.pdf")
