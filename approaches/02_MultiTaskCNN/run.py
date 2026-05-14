"""
Approach 2 — MultiTaskCNN  (classifier, not a generator)

  Generate only (default):
      python run.py
      → needs checkpoints/checkpoint.pt  (train in Colab first)

  Train then generate (Colab / GPU):
      python run.py --train --epochs 100
"""
import os, sys, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, ImageDraw
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dataset import load_all_writers, CHAR_TO_LABEL
from model   import MultiTaskCNN

import torchvision.transforms as T

DATA_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', 'Data', 'Writers_pngs'))
CKPT_DIR  = os.path.join(_HERE, 'checkpoints')
CKPT_PATH = os.path.join(CKPT_DIR, 'checkpoint.pt')
OUT_DIR   = os.path.join(_HERE, 'outputs')

LABEL_TO_CHAR = {v: k for k, v in CHAR_TO_LABEL.items()}
_tf = T.Compose([T.Grayscale(1), T.Resize((128, 128)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
SKIP = {'Writers_Zip', 'output_preview', '__pycache__'}


class WriterDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, char_label, writer_id = self.records[i]
        return _tf(Image.open(path)), char_label, writer_id


def load_data():
    raw         = load_all_writers(DATA_ROOT)
    writer_dirs = sorted(e.name for e in os.scandir(DATA_ROOT) if e.is_dir() and e.name not in SKIP)
    writer_map  = {name: i for i, name in enumerate(writer_dirs)}
    records = [(p, cl, writer_map.get(os.path.basename(os.path.dirname(p)), 0)) for p, cl in raw]
    return records, writer_dirs


def train(epochs, device):
    print(f'Loading data from {DATA_ROOT} ...')
    records, writer_names = load_data()
    dataset = WriterDataset(records)
    val_n   = max(1, int(len(dataset) * 0.15))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, 32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   32, shuffle=False, num_workers=0)

    model     = MultiTaskCNN(num_writers=len(writer_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, char_labels, writer_labels in train_loader:
            imgs, char_labels, writer_labels = imgs.to(device), char_labels.to(device), writer_labels.to(device)
            char_out, writer_out = model(imgs)
            loss = 0.4 * criterion(char_out, char_labels) + 0.6 * criterion(writer_out, writer_labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, char_labels, _ in val_loader:
                imgs, char_labels = imgs.to(device), char_labels.to(device)
                char_out, _ = model(imgs)
                correct += (char_out.argmax(1) == char_labels).sum().item()
                total   += imgs.size(0)
        print(f'  Epoch {ep:3d}/{epochs}  char_acc={correct/total:.3f}')

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'model': model.state_dict(), 'num_writers': len(writer_names),
                'writer_names': writer_names}, CKPT_PATH)
    print(f'Checkpoint -> {CKPT_PATH}')
    return model, writer_names


def load_ckpt(device):
    ck    = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = MultiTaskCNN(num_writers=ck['num_writers']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    return model, ck['writer_names']


def generate_grid(model, writer_names, device):
    records, _ = load_data()
    by_char = {}
    for path, char_label, writer_id in records:
        by_char.setdefault(char_label, []).append((path, writer_id))

    samples = []
    for char_label in sorted(by_char)[:20]:
        for path, writer_id in by_char[char_label][:4]:
            samples.append((path, char_label, writer_id))

    CELL, COLS = 144, 20
    ROWS = (len(samples) + COLS - 1) // COLS
    grid = Image.new('L', (COLS * CELL, ROWS * CELL), 240)
    draw = ImageDraw.Draw(grid)

    for i, (path, true_label, _) in enumerate(samples):
        img_pil = Image.open(path).convert('L').resize((128, 128))
        img_t   = _tf(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            char_out, writer_out = model(img_t)
        pred_char   = LABEL_TO_CHAR.get(char_out.argmax(1).item(), '?')
        pred_writer = writer_names[writer_out.argmax(1).item()] if writer_names else '?'
        true_char   = LABEL_TO_CHAR.get(true_label, '?')

        r, c = divmod(i, COLS)
        x, y = c * CELL, r * CELL
        grid.paste(img_pil, (x, y))
        draw.text((x + 2, y + 128), f'T:{true_char} P:{pred_char}', fill=0)

    os.makedirs(OUT_DIR, exist_ok=True)
    grid.save(os.path.join(OUT_DIR, 'predictions_grid.png'))
    print(f'Saved predictions_grid.png -> {OUT_DIR}/')


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
        print('Train in Colab first:  open approaches/02_MultiTaskCNN/Colab_MultiTaskCNN.ipynb')
        print('Then place checkpoint.pt in approaches/02_MultiTaskCNN/checkpoints/')
        sys.exit(1)

    generate_grid(model, writer_names, device)
