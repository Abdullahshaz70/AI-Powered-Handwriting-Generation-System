"""
Approach 5 — CharNet (current, has a trained checkpoint)
Generates a-z A-Z 0-9 PNGs immediately using the existing checkpoint.
Also trains from scratch if you want to retrain.

Run from this folder:
    python run.py                   # uses existing checkpoint
    python run.py --retrain --epochs 100
"""
import os, sys, argparse
import torch
from PIL import Image
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from generate import load_model, generate_char
from data     import CHAR_TO_IDX, CANVAS_SIZE

from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

# The trained checkpoint is at the project root checkpoints/
ROOT_CKPT  = os.path.normpath(os.path.join(_HERE, '..', '..', 'checkpoints', 'style_net.pt'))
OUT_DIR    = os.path.join(_HERE, 'outputs')
LOCAL_CKPT = os.path.join(OUT_DIR, 'style_net.pt')   # saved if retrained

DATA_ROOT  = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'bezier_labels.npy'))


def retrain(epochs, device):
    from data  import load_all_data, BezierDataset
    from model import CharNet

    print(f'Loading data (builds bezier_labels.npy cache on first run) ...')
    records, _ = load_all_data(DATA_ROOT, cache_path=CACHE_PATH)
    dataset    = BezierDataset(records)
    val_n      = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, 64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   64, shuffle=False, num_workers=0)

    model     = CharNet(num_chars=62).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for ep in range(1, epochs + 1):
        model.train()
        for char_t, cat_t, bezier_labels in train_loader:
            char_t, cat_t, bezier_labels = char_t.to(device), cat_t.to(device), bezier_labels.to(device)
            loss = F.mse_loss(model(char_t, cat_t), bezier_labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for char_t, cat_t, bezier_labels in val_loader:
                char_t, cat_t, bezier_labels = char_t.to(device), cat_t.to(device), bezier_labels.to(device)
                val_loss += F.mse_loss(model(char_t, cat_t), bezier_labels).item()
        scheduler.step()
        print(f'  Epoch {ep:3d}/{epochs}  val_MSE={val_loss/len(val_loader):.6f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'epoch': epochs, 'val_loss': val_loss / len(val_loader)},
               LOCAL_CKPT)
    print(f'Checkpoint -> {LOCAL_CKPT}')
    return LOCAL_CKPT


def generate_pngs(ckpt_path, device):
    model, ckpt, device = load_model(ckpt_path=ckpt_path, device=device)
    print(f'Loaded checkpoint: epoch={ckpt.get("epoch","?")}, val_MSE={ckpt.get("val_loss",0):.6f}')

    chars = [c for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
             if c in CHAR_TO_IDX]
    os.makedirs(OUT_DIR, exist_ok=True)
    def fname(ch):
        if ch.isupper():   return f'uc_{ch}.png'
        if ch.isdigit():   return f'digit_{ch}.png'
        return f'lc_{ch}.png'

    imgs = []
    for ch in chars:
        arr = generate_char(model, ch, device, noise_scale=0.018)
        imgs.append(arr)
        Image.fromarray(arr).save(os.path.join(OUT_DIR, fname(ch)))

    cols, H, W = 10, CANVAS_SIZE, CANVAS_SIZE
    rows = (len(imgs) + cols - 1) // cols
    grid = np.full((rows*(H+4), cols*(W+4)), 240, np.uint8)
    for i, arr in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r*(H+4):r*(H+4)+H, c*(W+4):c*(W+4)+W] = arr
    Image.fromarray(grid).save(os.path.join(OUT_DIR, 'grid.png'))
    print(f'Saved {len(imgs)} character PNGs + grid.png -> {OUT_DIR}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', action='store_true', help='Retrain from scratch instead of using existing checkpoint')
    ap.add_argument('--epochs',  type=int, default=100, help='Epochs when retraining (default 100)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.retrain:
        ckpt_path = retrain(args.epochs, device)
    elif os.path.exists(LOCAL_CKPT):
        print(f'Local checkpoint found: {LOCAL_CKPT}')
        ckpt_path = LOCAL_CKPT
    elif os.path.exists(ROOT_CKPT):
        print(f'Using project checkpoint: {ROOT_CKPT}')
        ckpt_path = ROOT_CKPT
    else:
        print('No checkpoint found. Run with --retrain to train first.')
        sys.exit(1)

    generate_pngs(ckpt_path, device)
