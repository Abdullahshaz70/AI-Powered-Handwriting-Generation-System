"""
Approach 1 — GAN

  Generate only (default):
      python run.py
      → needs checkpoints/checkpoint.pt  (train in Colab first)

  Train then generate (Colab / GPU):
      python run.py --train --epochs 100
"""
import os, sys, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset       import CharDataset, load_all_writers, CHAR_TO_LABEL
from encoder       import StyleEncoder
from generator     import CharacterGenerator
from discriminator import Discriminator

DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
CKPT_DIR  = os.path.join(_HERE, 'checkpoints')
CKPT_PATH = os.path.join(CKPT_DIR, 'checkpoint.pt')
OUT_DIR   = os.path.join(_HERE, 'outputs')


def train(epochs, device):
    print(f'Loading data from {DATA_ROOT} ...')
    loader = DataLoader(CharDataset(load_all_writers(DATA_ROOT)), 32, shuffle=True, num_workers=0)

    enc  = StyleEncoder().to(device)
    gen  = CharacterGenerator().to(device)
    disc = Discriminator().to(device)
    opt_g = optim.Adam(list(enc.parameters()) + list(gen.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    adv   = nn.BCEWithLogitsLoss()
    cls_  = nn.CrossEntropyLoss()
    l1    = nn.L1Loss()

    for ep in range(1, epochs + 1):
        g_sum = d_sum = n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            style = enc(imgs)
            fakes = gen(style.detach(), labels)
            r_rf, r_cls = disc(imgs, labels)
            f_rf, _     = disc(fakes.detach(), labels)
            d_loss = (adv(r_rf, torch.ones_like(r_rf))
                    + adv(f_rf, torch.zeros_like(f_rf))
                    + cls_(r_cls, labels))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            style = enc(imgs)
            fakes = gen(style, labels)
            f_rf, f_cls = disc(fakes, labels)
            g_loss = (adv(f_rf, torch.ones_like(f_rf))
                    + cls_(f_cls, labels)
                    + 100.0 * l1(fakes, imgs))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            b = imgs.size(0); g_sum += g_loss.item() * b; d_sum += d_loss.item() * b; n += b

        print(f'  Epoch {ep:3d}/{epochs}  G={g_sum/n:.4f}  D={d_sum/n:.4f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'enc': enc.state_dict(), 'gen': gen.state_dict(), 'epochs': epochs}, CKPT_PATH)
    print(f'Checkpoint -> {CKPT_PATH}')
    return enc, gen


def load_ckpt(device):
    ck  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    enc = StyleEncoder().to(device);       enc.load_state_dict(ck['enc']);  enc.eval()
    gen = CharacterGenerator().to(device); gen.load_state_dict(ck['gen']); gen.eval()
    return enc, gen


_infer_tf = T.Compose([
    T.Grayscale(1),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)),
])


def generate(enc, gen, device):
    # Group all dataset paths by character label
    data     = load_all_writers(DATA_ROOT)
    by_label = {}
    for path, label in data:
        by_label.setdefault(label, []).append(path)

    def fname(ch):
        if ch.isupper():  return f'uc_{ch}.png'
        if ch.isdigit():  return f'digit_{ch}.png'
        return f'lc_{ch}.png'

    chars = sorted(CHAR_TO_LABEL.items(), key=lambda x: x[1])
    imgs  = []
    with torch.no_grad():
        for char, idx in chars:
            # Use a real image of THIS character as the reference (no augmentation)
            paths = by_label.get(idx, [])
            if not paths:
                continue
            ref = _infer_tf(Image.open(paths[0])).unsqueeze(0).to(device)
            style = enc(ref)
            out = gen(style, torch.tensor([idx], device=device))
            arr = ((out.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            imgs.append((char, arr))

    os.makedirs(OUT_DIR, exist_ok=True)
    for char, arr in imgs:
        Image.fromarray(arr).save(os.path.join(OUT_DIR, fname(char)))

    cols, H, W = 10, 128, 128
    rows = (len(imgs) + cols - 1) // cols
    grid = np.full((rows*(H+4), cols*(W+4)), 240, np.uint8)
    for i, (_, arr) in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r*(H+4):r*(H+4)+H, c*(W+4):c*(W+4)+W] = arr
    Image.fromarray(grid).save(os.path.join(OUT_DIR, 'grid.png'))
    print(f'Saved {len(imgs)} PNGs + grid.png -> {OUT_DIR}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train',  action='store_true', help='Train the model (use on GPU / Colab)')
    ap.add_argument('--epochs', type=int, default=100, help='Training epochs (default 100)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.train:
        enc, gen = train(args.epochs, device)
    elif os.path.exists(CKPT_PATH):
        print(f'Checkpoint found: {CKPT_PATH}')
        enc, gen = load_ckpt(device)
    else:
        print(f'No checkpoint at {CKPT_PATH}')
        print('Train in Colab first:  open approaches/01_GAN/Colab_GAN.ipynb')
        print('Then place checkpoint.pt in approaches/01_GAN/checkpoints/')
        sys.exit(1)

    generate(enc, gen, device)
