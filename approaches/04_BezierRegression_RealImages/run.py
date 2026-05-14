"""
Approach 4 — Bézier Regression from Real Images

  Generate only (default):
      python run.py
      → needs checkpoints/style_net.pt  (train in Colab first)

  Train then generate (Colab / GPU):
      python run.py --train --epochs 100
"""
import os, sys, argparse, random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from data    import load_all_data, HandwritingDataset, CHAR_TO_IDX, scale_to_canvas, CANVAS_SIZE
from model   import HandwritingCNN
from bezier  import label_to_curves, add_variation, draw_bezier

DATA_ROOT  = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'bezier_labels.npy'))
CKPT_DIR   = os.path.join(_HERE, 'checkpoints')
CKPT_PATH  = os.path.join(CKPT_DIR, 'style_net.pt')
OUT_DIR    = os.path.join(_HERE, 'outputs')

SKIP_DIRS = {'Writers_Zip', 'output_preview', '__pycache__'}


def _load_writer_refs(data_root):
    refs = {}
    dirs = sorted([e for e in os.scandir(data_root) if e.is_dir() and e.name not in SKIP_DIRS],
                  key=lambda e: e.name)
    for wid, entry in enumerate(dirs):
        imgs = [Image.open(os.path.join(entry.path, f)).convert('L')
                for f in sorted(os.listdir(entry.path)) if f.lower().endswith('.png')]
        refs[wid] = imgs
    return refs


def _img_to_tensor(pil_img, device):
    arr = np.array(pil_img.resize((CANVAS_SIZE, CANVAS_SIZE)))
    arr = scale_to_canvas(np.where(arr < 128, 0, 255).astype(np.uint8), target_fill=0.80)
    t   = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
    return t.to(device)


def train(epochs, device):
    print('Loading data (builds bezier_labels.npy cache on first run — a few minutes) ...')
    records, writer_names = load_all_data(DATA_ROOT, cache_path=CACHE_PATH)
    dataset  = HandwritingDataset(records)
    val_n    = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, 32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   32, shuffle=False, num_workers=0)

    model     = HandwritingCNN(num_chars=62).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, char_idxs, bezier_labels in train_loader:
            imgs, char_idxs, bezier_labels = imgs.to(device), char_idxs.to(device), bezier_labels.to(device)
            loss = F.mse_loss(model(imgs, char_idxs), bezier_labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, char_idxs, bezier_labels in val_loader:
                imgs, char_idxs, bezier_labels = imgs.to(device), char_idxs.to(device), bezier_labels.to(device)
                val_loss += F.mse_loss(model(imgs, char_idxs), bezier_labels).item()
        scheduler.step()
        print(f'  Epoch {ep:3d}/{epochs}  val_MSE={val_loss/len(val_loader):.6f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'num_writers': len(writer_names),
                'writer_names': writer_names, 'epoch': epochs,
                'val_loss': val_loss / len(val_loader)}, CKPT_PATH)
    print(f'Checkpoint -> {CKPT_PATH}')
    return model, writer_names


def load_ckpt(device):
    ck    = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = HandwritingCNN(num_chars=62).to(device)
    model.load_state_dict(ck['model_state']); model.eval()
    return model, ck['writer_names']


def _generate_char(model, char, ref_img, device):
    tensor   = _img_to_tensor(ref_img, device)
    char_idx = torch.tensor([CHAR_TO_IDX[char]], device=device)
    with torch.no_grad():
        label = model(tensor, char_idx).squeeze().cpu().numpy()
    curves = label_to_curves(label)
    varied = add_variation(curves, noise_scale=0.018)
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
    return draw_bezier(canvas, varied, thickness=3)


def generate_pngs(model, writer_names, device):
    refs  = _load_writer_refs(DATA_ROOT)
    chars = [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
             if c in CHAR_TO_IDX]

    def fname(writer_idx, ch):
        if ch.isupper():  return f'writer{writer_idx}_uc_{ch}.png'
        if ch.isdigit():  return f'writer{writer_idx}_digit_{ch}.png'
        return f'writer{writer_idx}_lc_{ch}.png'

    os.makedirs(OUT_DIR, exist_ok=True)
    for writer_idx, writer_name in enumerate(writer_names):
        writer_refs = refs.get(writer_idx, [])
        if not writer_refs:
            continue

        imgs = []
        for ch in chars:
            ref_img = random.choice(writer_refs)
            arr     = _generate_char(model, ch, ref_img, device)
            imgs.append(arr)
            Image.fromarray(arr).save(os.path.join(OUT_DIR, fname(writer_idx, ch)))

        cols, H, W = 10, CANVAS_SIZE, CANVAS_SIZE
        rows = (len(imgs) + cols - 1) // cols
        grid = np.full((rows*(H+4), cols*(W+4)), 240, np.uint8)
        for i, arr in enumerate(imgs):
            r, c = divmod(i, cols)
            grid[r*(H+4):r*(H+4)+H, c*(W+4):c*(W+4)+W] = arr
        Image.fromarray(grid).save(os.path.join(OUT_DIR, f'grid_writer{writer_idx}.png'))
        print(f'  {writer_name}: {len(imgs)} chars + grid saved')

    print(f'All PNGs -> {OUT_DIR}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train',  action='store_true', help='Train the model (use on GPU / Colab)')
    ap.add_argument('--epochs', type=int, default=100, help='Training epochs (default 100)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.train:
        model, writer_names = train(args.epochs, device)
    elif os.path.exists(CKPT_PATH):
        print(f'Checkpoint found: {CKPT_PATH}')
        model, writer_names = load_ckpt(device)
    else:
        print(f'No checkpoint at {CKPT_PATH}')
        print('Train in Colab first:  open approaches/04_BezierRegression_RealImages/Colab_BezierReg.ipynb')
        print('Then place style_net.pt in approaches/04_BezierRegression_RealImages/checkpoints/')
        sys.exit(1)

    generate_pngs(model, writer_names, device)
