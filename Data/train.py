import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg16, VGG16_Weights

from dataset import CharDataset, load_all_writers
from encoder import StyleEncoder
from generator import CharacterGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(_HERE, "Writers_pngs")
CHECKPOINT_DIR = os.path.join(_HERE, "..", "checkpoints")
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
PIXEL_LOSS_WEIGHT = 1.0
PERCEPTUAL_LOSS_WEIGHT = 0.1
SAVE_EVERY = 5
RESUME_FROM = os.path.join(_HERE, "..", "checkpoints", "checkpoint_epoch_005.pt")


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(DEVICE)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, generated, target):
        gen_rgb = generated.repeat(1, 3, 1, 1)
        tgt_rgb = target.repeat(1, 3, 1, 1)
        gen_feats = self.vgg(gen_rgb)
        tgt_feats = self.vgg(tgt_rgb)
        return self.mse(gen_feats, tgt_feats)


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


def run_epoch(encoder, generator, loader, optimizer, pixel_criterion, perceptual_criterion, training):
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

            pixel_loss = pixel_criterion(generated, images)
            perceptual_loss = perceptual_criterion(generated, images)
            loss = PIXEL_LOSS_WEIGHT * pixel_loss + PERCEPTUAL_LOSS_WEIGHT * perceptual_loss

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

    pixel_criterion = nn.MSELoss()
    perceptual_criterion = PerceptualLoss()

    start_epoch = 0
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        start_epoch = load_checkpoint(encoder, generator, optimizer, RESUME_FROM)

    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        train_loss = run_epoch(encoder, generator, train_loader, optimizer, pixel_criterion, perceptual_criterion, training=True)
        val_loss = run_epoch(encoder, generator, val_loader, optimizer, pixel_criterion, perceptual_criterion, training=False)

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