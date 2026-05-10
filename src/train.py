"""
Training loop for the HandwritingCNN regression model.

Run from the project root:
    python src/train.py
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from data  import load_all_data, HandwritingDataset, NUM_CHARS
from model import HandwritingCNN

# ── config ───────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(_HERE, '..', 'Data', 'Writers_pngs')
CKPT_DIR    = os.path.join(_HERE, '..', 'checkpoints')
NUM_EPOCHS  = 60
BATCH_SIZE  = 32
LR          = 1e-3
LR_MIN      = 1e-5
VAL_SPLIT   = 0.15


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = n = 0
    with torch.set_grad_enabled(train):
        for imgs, char_idx, labels in loader:
            imgs, char_idx, labels = (
                imgs.to(device), char_idx.to(device), labels.to(device)
            )
            pred = model(imgs, char_idx)
            loss = criterion(pred, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total_loss / n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading dataset...')
    records, writer_names = load_all_data(DATA_ROOT, cache_labels=True)

    dataset  = HandwritingDataset(records, augment=True)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type == 'cuda')
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    print(f'Train: {len(train_ds)}  |  Val: {len(val_ds)}\n')

    model     = HandwritingCNN(num_chars=NUM_CHARS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val = float('inf')

    header = f"{'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>9} | {'LR':>8}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        tl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f'{epoch:>5} | {tl:>10.5f} | {vl:>9.5f} | {lr:.2e}')

        if vl < best_val:
            best_val = vl
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'val_loss':     vl,
                'writer_names': writer_names,
                'num_chars':    NUM_CHARS,
            }, os.path.join(CKPT_DIR, 'generator.pt'))
            print(f'  >> generator.pt saved (val_loss={vl:.5f})')

    print(f'\nBest val MSE: {best_val:.5f}')
    print(f'Checkpoint: {os.path.join(CKPT_DIR, "generator.pt")}')


if __name__ == '__main__':
    main()
