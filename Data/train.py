import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CharDataset, load_all_writers
from encoder import StyleEncoder
from generator import CharacterGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(_HERE, "Writers_pngs")
CHECKPOINT_DIR = os.path.join(_HERE, "..", "checkpoints")
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.10
L1_LOSS_WEIGHT = 1.0
SSIM_LOSS_WEIGHT = 1.0
SAVE_EVERY = 5
RESUME_FROM = None


class SSIMLoss(nn.Module):
    """Structural Similarity loss — penalises blurry outputs unlike MSE."""
    def __init__(self, window_size=11):
        super().__init__()
        self.ws = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        ws, pad = self.ws, self.ws // 2
        mu1 = F.avg_pool2d(pred,   ws, 1, pad)
        mu2 = F.avg_pool2d(target, ws, 1, pad)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        s1  = F.avg_pool2d(pred   * pred,   ws, 1, pad) - mu1_sq
        s2  = F.avg_pool2d(target * target, ws, 1, pad) - mu2_sq
        s12 = F.avg_pool2d(pred   * target, ws, 1, pad) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * s12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2))
        return 1 - ssim_map.mean()


def save_checkpoint(encoder, generator, optimizer, epoch, train_loss, val_loss):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "encoder_state": encoder.state_dict(),
        "generator_state": generator.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(encoder, generator, optimizer, path):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    generator.load_state_dict(ckpt["generator_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def run_epoch(encoder, generator, loader, optimizer, l1_criterion, ssim_criterion, training):
    encoder.train(training)
    generator.train(training)

    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            style = encoder(images)
            generated = generator(style, labels)

            l1_loss   = l1_criterion(generated, images)
            ssim_loss = ssim_criterion(generated, images)
            loss = L1_LOSS_WEIGHT * l1_loss + SSIM_LOSS_WEIGHT * ssim_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    print(f"Device: {DEVICE}")

    data_list = load_all_writers(DATA_FOLDER)
    print(f"Total samples loaded: {len(data_list)}")

    full_dataset = CharDataset(data_list)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size}")

    encoder = StyleEncoder().to(DEVICE)
    generator = CharacterGenerator().to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    l1_criterion   = nn.L1Loss()
    ssim_criterion = SSIMLoss().to(DEVICE)

    start_epoch = 0
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        start_epoch = load_checkpoint(encoder, generator, optimizer, RESUME_FROM)

    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        train_loss = run_epoch(encoder, generator, train_loader, optimizer, l1_criterion, ssim_criterion, training=True)
        val_loss   = run_epoch(encoder, generator, val_loader,   optimizer, l1_criterion, ssim_criterion, training=False)

        scheduler.step()

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "generator_state": generator.state_dict(),
                "val_loss": val_loss,
            }, best_path)
            print(f"  Best model updated (val_loss={val_loss:.5f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(encoder, generator, optimizer, epoch, train_loss, val_loss)

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == "__main__":
    main()