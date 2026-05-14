"""
Approach 3 — Font-to-Handwriting CNN

  Generate only (default):
      python run.py
      → needs checkpoints/style_net.pt  (train in Colab first)

  Train then generate (Colab / GPU):
      python run.py --train --epochs 80
"""
import os, sys, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from data        import load_all_data, HandwritingDataset, CHAR_TO_IDX
from model       import HandwritingStyleNet
from generate    import generate_char

DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
CKPT_DIR  = os.path.join(_HERE, 'checkpoints')
CKPT_PATH = os.path.join(CKPT_DIR, 'style_net.pt')
OUT_DIR   = os.path.join(_HERE, 'outputs')


def train(epochs, device):
    print(f'Loading data from {DATA_ROOT} ...')
    records, writer_names = load_all_data(DATA_ROOT)
    dataset  = HandwritingDataset(records, augment=True)
    val_n    = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, 16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   16, shuffle=False, num_workers=0)

    model     = HandwritingStyleNet(num_writers=len(writer_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for ep in range(1, epochs + 1):
        model.train()
        for font_t, real_t, _, writer_idx in train_loader:
            font_t, real_t = font_t.to(device), real_t.to(device)
            loss = F.l1_loss(model(font_t, writer_idx.to(device)), real_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for font_t, real_t, _, writer_idx in val_loader:
                font_t, real_t = font_t.to(device), real_t.to(device)
                val_loss += F.l1_loss(model(font_t, writer_idx.to(device)), real_t).item()
        scheduler.step()
        print(f'  Epoch {ep:3d}/{epochs}  val_L1={val_loss/len(val_loader):.4f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'num_writers': len(writer_names),
                'writer_names': writer_names, 'epoch': epochs, 'val_loss': val_loss/len(val_loader)},
               CKPT_PATH)
    print(f'Checkpoint -> {CKPT_PATH}')
    return model, writer_names


def load_ckpt(device):
    ck    = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = HandwritingStyleNet(num_writers=ck['num_writers']).to(device)
    model.load_state_dict(ck['model_state']); model.eval()
    return model, ck['writer_names']


def generate_pngs(model, writer_names, device):
    chars = [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
             if c in CHAR_TO_IDX]

    def fname(writer_idx, ch):
        if ch.isupper():  return f'writer{writer_idx}_uc_{ch}.png'
        if ch.isdigit():  return f'writer{writer_idx}_digit_{ch}.png'
        return f'writer{writer_idx}_lc_{ch}.png'

    os.makedirs(OUT_DIR, exist_ok=True)
    for writer_idx, writer_name in enumerate(writer_names):
        imgs = []
        for ch in chars:
            arr = generate_char(model, ch, writer_idx, device)
            imgs.append(arr)
            Image.fromarray(arr).save(os.path.join(OUT_DIR, fname(writer_idx, ch)))

        cols, H, W = 10, 128, 128
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
    ap.add_argument('--epochs', type=int, default=80, help='Training epochs (default 80)')
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
        print('Train in Colab first:  open approaches/03_FontToHandwriting_CNN/Colab_FontCNN.ipynb')
        print('Then place style_net.pt in approaches/03_FontToHandwriting_CNN/checkpoints/')
        sys.exit(1)

    generate_pngs(model, writer_names, device)
