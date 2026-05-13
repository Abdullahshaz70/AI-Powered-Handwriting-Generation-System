"""
Approach 3 — Font-to-Handwriting CNN
Trains if no checkpoint found, then generates a-z A-Z 0-9 PNGs for each writer.

Run from this folder:
    python run.py
    python run.py --epochs 40
"""
import os, sys, argparse
import torch
from PIL import Image
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from data        import load_all_data, HandwritingDataset, CHAR_TO_IDX, IDX_TO_CHAR
from model       import HandwritingStyleNet
from font_render import render_char
from generate    import generate_char, stitch_word

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
OUT_DIR   = os.path.join(_HERE, 'outputs')
CKPT_PATH = os.path.join(OUT_DIR, 'style_net.pt')

_transform = T.Compose([T.Grayscale(1), T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])


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

    import torch.nn.functional as F
    for ep in range(1, epochs + 1):
        model.train()
        for font_t, real_t, char_idx, writer_idx in train_loader:
            font_t, real_t = font_t.to(device), real_t.to(device)
            writer_idx     = writer_idx.to(device)
            out  = model(font_t, writer_idx)
            loss = F.l1_loss(out, real_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for font_t, real_t, char_idx, writer_idx in val_loader:
                font_t, real_t = font_t.to(device), real_t.to(device)
                writer_idx = writer_idx.to(device)
                val_loss += F.l1_loss(model(font_t, writer_idx), real_t).item()
        scheduler.step()
        print(f'  Epoch {ep:3d}/{epochs}  val_L1={val_loss/len(val_loader):.4f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'num_writers': len(writer_names),
                'writer_names': writer_names, 'epoch': epochs}, CKPT_PATH)
    print(f'Checkpoint -> {CKPT_PATH}')
    return model, writer_names


def load_ckpt(device):
    ck    = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = HandwritingStyleNet(num_writers=ck['num_writers']).to(device)
    model.load_state_dict(ck['model_state']); model.eval()
    return model, ck['writer_names']


def generate_pngs(model, writer_names, device):
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    os.makedirs(OUT_DIR, exist_ok=True)

    for writer_idx, writer_name in enumerate(writer_names):
        def fname(ch):
            if ch.isupper():  return f'writer{writer_idx}_uc_{ch}.png'
            if ch.isdigit():  return f'writer{writer_idx}_digit_{ch}.png'
            return f'writer{writer_idx}_lc_{ch}.png'

        imgs = []
        for ch in chars:
            if ch not in CHAR_TO_IDX:
                continue
            arr = generate_char(model, ch, writer_idx, device)
            imgs.append(arr)
            Image.fromarray(arr).save(os.path.join(OUT_DIR, fname(ch)))

        # grid per writer
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
    ap.add_argument('--epochs', type=int, default=20,
                    help='Training epochs if no checkpoint found (default 20; try 80 for quality)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if os.path.exists(CKPT_PATH):
        print(f'Checkpoint found — loading {CKPT_PATH}')
        model, writer_names = load_ckpt(device)
    else:
        print(f'No checkpoint — training for {args.epochs} epochs ...')
        model, writer_names = train(args.epochs, device)

    generate_pngs(model, writer_names, device)
