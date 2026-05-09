import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CharDataset, load_dataset, CHAR_TO_LABEL
from model import MultiTaskCNN

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "Writers_pngs")
CKPT_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
NUM_EPOCHS  = 100
BATCH_SIZE  = 32
LR          = 1e-3
VAL_SPLIT   = 0.15
STEP_SIZE   = 20
GAMMA       = 0.5
SKIP_DIRS   = {"Writers_Zip", "output_preview"}


# ── dataset with writer labels ─────────────────────────────────────────────────

def load_all_writers_with_id(root_folder):
    all_data, writer_names = [], []
    for entry in sorted(os.scandir(root_folder), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        samples = load_dataset(entry.path)
        if not samples:
            continue
        writer_id = len(writer_names)
        writer_names.append(entry.name)
        for img_path, char_label in samples:
            all_data.append((img_path, char_label, writer_id))
        print(f"  {entry.name}: {len(samples)} samples  (writer_id={writer_id})")
    print(f"Total: {len(all_data)} samples, {len(writer_names)} writers")
    return all_data, writer_names


class MultiTaskCharDataset(CharDataset):
    """CharDataset extended to return (image, char_label, writer_id)."""

    def __getitem__(self, i):
        img_path, char_label, writer_id = self.data[i]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, char_label, writer_id


# ── helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()


def run_epoch(model, loader, char_criterion, writer_criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = char_correct = writer_correct = n = 0
    with torch.set_grad_enabled(train):
        for images, char_labels, writer_labels in loader:
            images        = images.to(device)
            char_labels   = char_labels.to(device)
            writer_labels = writer_labels.to(device)

            char_logits, writer_logits = model(images)
            loss = 0.4 * char_criterion(char_logits, char_labels) + \
                   0.6 * writer_criterion(writer_logits, writer_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            total_loss    += loss.item() * bs
            char_correct  += (char_logits.argmax(1) == char_labels).sum().item()
            writer_correct+= (writer_logits.argmax(1) == writer_labels).sum().item()
            n += bs

    return total_loss / n, char_correct / n, writer_correct / n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if torch.cuda.is_available() else 0
    print(f"Device: {device}  |  num_workers: {num_workers}")

    print("\nLoading dataset...")
    all_data, writer_names = load_all_writers_with_id(DATA_ROOT)
    num_writers = len(writer_names)

    dataset   = MultiTaskCharDataset(all_data)
    val_size  = int(len(dataset) * VAL_SPLIT)
    train_size= len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\nTrain: {len(train_ds)}  |  Val: {len(val_ds)}")
    print(f"Writers: {writer_names}\n")

    model     = MultiTaskCNN(num_writers=num_writers).to(device)
    char_criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    writer_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_char_acc = 0.0

    header = f"{'Epoch':>5} | {'TrainChar':>9} | {'TrainWriter':>11} | {'ValChar':>7} | {'ValWriter':>9} | {'LR':>8}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, NUM_EPOCHS + 1):
        _, tr_char, tr_writer = run_epoch(model, train_loader, char_criterion, writer_criterion, optimizer, device, train=True)
        _, vl_char, vl_writer = run_epoch(model, val_loader,   char_criterion, writer_criterion, optimizer, device, train=False)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>5} | {tr_char*100:>8.2f}% | {tr_writer*100:>10.2f}% | "
              f"{vl_char*100:>6.2f}% | {vl_writer*100:>8.2f}% | {current_lr:.2e}")

        if vl_char > best_char_acc:
            best_char_acc = vl_char
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "char_acc":     vl_char,
                "writer_acc":   vl_writer,
                "writer_names": writer_names,
                "num_writers":  num_writers,
            }, os.path.join(CKPT_DIR, "best_model.pt"))
            print(f"  >> best_model.pt saved (val_char={vl_char*100:.2f}%)")

    print(f"\nTraining complete. Best val char accuracy: {best_char_acc*100:.2f}%")
    print(f"Checkpoint: {os.path.join(CKPT_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
