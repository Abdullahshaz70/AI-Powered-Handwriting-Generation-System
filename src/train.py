"""
Training loop for HandwritingStyleNet (font → handwriting image translation).

Loss: L1 (MAE) — produces sharper outputs than MSE for image translation.
Writer style is injected via an embedding at the bottleneck.

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
from model import HandwritingStyleNet

# ── config ───────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(_HERE, '..', 'Data', 'Writers_pngs')
CKPT_DIR   = os.path.join(_HERE, '..', 'checkpoints')
NUM_EPOCHS = 80
BATCH_SIZE = 16
LR         = 2e-4
VAL_SPLIT  = 0.10


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\nLoading dataset...')
    records, writer_names = load_all_data(DATA_ROOT)
    num_writers = len(writer_names)

    dataset  = HandwritingDataset(records, augment=True)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type == 'cuda')
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    print(f'Train: {len(train_ds)}  |  Val: {len(val_ds)}\n')

    model     = HandwritingStyleNet(num_writers=num_writers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {total_params:,}\n')

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val = float('inf')

    header = f"{'Epoch':>5} | {'Train L1':>9} | {'Val L1':>8} | {'LR':>8}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tl = tn = 0
        for font_img, real_img, _, writer_idx in train_loader:
            font_img   = font_img.to(device)
            real_img   = real_img.to(device)
            writer_idx = writer_idx.to(device)

            pred = model(font_img, writer_idx)
            loss = F.l1_loss(pred, real_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * font_img.size(0)
            tn += font_img.size(0)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        vl = vn = 0
        with torch.no_grad():
            for font_img, real_img, _, writer_idx in val_loader:
                font_img   = font_img.to(device)
                real_img   = real_img.to(device)
                writer_idx = writer_idx.to(device)
                pred = model(font_img, writer_idx)
                vl  += F.l1_loss(pred, real_img).item() * font_img.size(0)
                vn  += font_img.size(0)

        scheduler.step()
        lr         = scheduler.get_last_lr()[0]
        train_loss = tl / tn
        val_loss   = vl / vn

        print(f'{epoch:>5} | {train_loss:>9.4f} | {val_loss:>8.4f} | {lr:.2e}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'val_loss':     val_loss,
                'writer_names': writer_names,
                'num_writers':  num_writers,
            }, os.path.join(CKPT_DIR, 'style_net.pt'))
            print(f'  >> style_net.pt saved (val_L1={val_loss:.4f})')

    print(f'\nBest val L1: {best_val:.4f}')
    print(f'Checkpoint : {os.path.join(CKPT_DIR, "style_net.pt")}')


if __name__ == '__main__':
    main()
