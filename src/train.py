"""
Training loop for HandwritingCNN (real image → Bézier curve regression).

Loss: MSE on 24 control-point coordinates in [0, 1].
The CNN sees a real handwriting image and predicts the Bézier skeleton curves
for that character — implicitly encoding each writer's stroke style.

Run from project root:
    python src/train.py
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from data  import load_all_data, HandwritingDataset
from model import HandwritingCNN

# ── config ───────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.normpath(os.path.join(_HERE, '..', 'Data', 'Writers_pngs'))
CACHE_PATH = os.path.normpath(os.path.join(_HERE, '..', 'Data', 'bezier_labels.npy'))
CKPT_DIR   = os.path.normpath(os.path.join(_HERE, '..', 'checkpoints'))
NUM_EPOCHS = 100
BATCH_SIZE = 32
LR         = 1e-3
VAL_SPLIT  = 0.10


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading dataset (first run builds label cache — may take a few minutes)...')
    records, writer_names = load_all_data(DATA_ROOT, cache_path=CACHE_PATH)

    dataset  = HandwritingDataset(records)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    nw = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=device.type == 'cuda')
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=device.type == 'cuda')
    print(f'Train: {len(train_ds)}  |  Val: {len(val_ds)}\n')

    model     = HandwritingCNN(num_chars=62).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}\n')

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val = float('inf')

    header = f"{'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>9} | {'LR':>8}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tl = tn = 0
        for img, char_idx, _, label in train_loader:
            img      = img.to(device)
            char_idx = char_idx.to(device)
            label    = label.to(device)

            pred = model(img, char_idx)
            loss = F.mse_loss(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * img.size(0)
            tn += img.size(0)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        vl = vn = 0
        with torch.no_grad():
            for img, char_idx, _, label in val_loader:
                img      = img.to(device)
                char_idx = char_idx.to(device)
                label    = label.to(device)
                pred     = model(img, char_idx)
                vl      += F.mse_loss(pred, label).item() * img.size(0)
                vn      += img.size(0)

        scheduler.step()
        lr         = scheduler.get_last_lr()[0]
        train_loss = tl / tn
        val_loss   = vl / vn

        print(f'{epoch:>5} | {train_loss:>10.6f} | {val_loss:>9.6f} | {lr:.2e}', flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'val_loss':     val_loss,
                'writer_names': writer_names,
                'num_writers':  len(writer_names),
            }, os.path.join(CKPT_DIR, 'style_net.pt'))
            print(f'  >> style_net.pt saved (val_MSE={val_loss:.6f})', flush=True)

    print(f'\nBest val MSE: {best_val:.6f}')
    print(f'Checkpoint  : {os.path.join(CKPT_DIR, "style_net.pt")}')


if __name__ == '__main__':
    main()
