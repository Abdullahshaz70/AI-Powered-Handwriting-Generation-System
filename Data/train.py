import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CharDataset, load_all_writers
from encoder import StyleEncoder
from generator import CharacterGenerator
from discriminator import Discriminator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER    = os.path.join(_HERE, "Writers_pngs")
CHECKPOINT_DIR = os.path.join(_HERE, "..", "checkpoints")

NUM_EPOCHS  = 100
BATCH_SIZE  = 32
LR_G        = 2e-4
LR_D        = 1e-4
BETA1       = 0.5
BETA2       = 0.999
L1_WEIGHT   = 100.0
SSIM_WEIGHT = 10.0
ADV_WEIGHT  = 1.0
CLS_WEIGHT  = 2.0     # auxiliary classifier weight — forces character distinctness
VAL_SPLIT   = 0.10
SAVE_EVERY  = 5
RESUME_FROM = None


class SSIMLoss(nn.Module):
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


def save_checkpoint(encoder, generator, discriminator, opt_g, opt_d, epoch, g_loss, val_loss):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch":               epoch,
        "encoder_state":       encoder.state_dict(),
        "generator_state":     generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "optimizer_g_state":   opt_g.state_dict(),
        "optimizer_d_state":   opt_d.state_dict(),
        "g_loss":              g_loss,
        "val_loss":            val_loss,
    }, path)
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(encoder, generator, discriminator, opt_g, opt_d, path):
    ckpt = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(ckpt["encoder_state"])
    generator.load_state_dict(ckpt["generator_state"])
    if "discriminator_state" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator_state"])
    if "optimizer_g_state" in ckpt:
        opt_g.load_state_dict(ckpt["optimizer_g_state"])
    if "optimizer_d_state" in ckpt:
        opt_d.load_state_dict(ckpt["optimizer_d_state"])
    print(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


def train_epoch(encoder, generator, discriminator, loader, opt_g, opt_d,
                l1_crit, ssim_crit, bce_crit, ce_crit):
    encoder.train()
    generator.train()
    discriminator.train()

    total_d = 0.0
    total_g = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        b = images.size(0)

        # ---- Discriminator step ----
        opt_d.zero_grad()
        with torch.no_grad():
            # Decouple style from content: shuffle so style of image[i] != target[i]
            perm  = torch.randperm(b, device=DEVICE)
            style = encoder(images[perm])
            fake  = generator(style, labels)

        rf_real, cls_real = discriminator(images, labels)
        rf_fake, cls_fake = discriminator(fake.detach(), labels)

        d_rf  = (bce_crit(rf_real, torch.full((b, 1), 0.9, device=DEVICE)) +
                 bce_crit(rf_fake, torch.zeros(b, 1, device=DEVICE))) * 0.5
        d_cls = (ce_crit(cls_real, labels) + ce_crit(cls_fake, labels)) * 0.5
        d_loss = d_rf + CLS_WEIGHT * d_cls
        d_loss.backward()
        opt_d.step()

        # ---- Generator + Encoder step ----
        opt_g.zero_grad()
        perm  = torch.randperm(b, device=DEVICE)
        style = encoder(images[perm])
        fake  = generator(style, labels)

        rf_fake, cls_fake = discriminator(fake, labels)
        adv_loss  = bce_crit(rf_fake, torch.ones(b, 1, device=DEVICE))
        cls_loss  = ce_crit(cls_fake, labels)   # must fool the character classifier
        l1_loss   = l1_crit(fake, images)
        ssim_loss = ssim_crit(fake, images)

        g_loss = (ADV_WEIGHT  * adv_loss +
                  CLS_WEIGHT  * cls_loss +
                  L1_WEIGHT   * l1_loss  +
                  SSIM_WEIGHT * ssim_loss)
        g_loss.backward()
        opt_g.step()

        total_d += d_loss.item() * b
        total_g += g_loss.item() * b

    n = len(loader.dataset)
    return total_d / n, total_g / n


def val_epoch(encoder, generator, loader, l1_crit, ssim_crit):
    encoder.eval()
    generator.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            style  = encoder(images)
            fake   = generator(style, labels)
            total += (L1_WEIGHT * l1_crit(fake, images) +
                      SSIM_WEIGHT * ssim_crit(fake, images)).item() * images.size(0)
    return total / len(loader.dataset)


def main():
    print(f"Device: {DEVICE}")

    data_list = load_all_writers(DATA_FOLDER)
    print(f"Total samples loaded: {len(data_list)}")

    full_dataset = CharDataset(data_list)
    val_size     = int(len(full_dataset) * VAL_SPLIT)
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    nw = 4 if DEVICE.type == "cuda" else 0
    pm = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pm)
    print(f"Train: {train_size} | Val: {val_size}")

    encoder       = StyleEncoder().to(DEVICE)
    generator     = CharacterGenerator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    opt_g = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=LR_G, betas=(BETA1, BETA2)
    )
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    scheduler_g = optim.lr_scheduler.StepLR(opt_g, step_size=30, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=30, gamma=0.5)

    l1_crit   = nn.L1Loss()
    ssim_crit = SSIMLoss().to(DEVICE)
    bce_crit  = nn.BCEWithLogitsLoss()
    ce_crit   = nn.CrossEntropyLoss()

    start_epoch = 0
    if RESUME_FROM and os.path.isfile(RESUME_FROM):
        start_epoch = load_checkpoint(encoder, generator, discriminator, opt_g, opt_d, RESUME_FROM)

    best_val_loss = float("inf")

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        d_loss, g_loss = train_epoch(
            encoder, generator, discriminator,
            train_loader, opt_g, opt_d,
            l1_crit, ssim_crit, bce_crit, ce_crit
        )
        val_loss = val_epoch(encoder, generator, val_loader, l1_crit, ssim_crit)

        scheduler_g.step()
        scheduler_d.step()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"D: {d_loss:.4f} | G: {g_loss:.4f} | "
            f"Val: {val_loss:.4f} | LR_G: {scheduler_g.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                "epoch":           epoch,
                "encoder_state":   encoder.state_dict(),
                "generator_state": generator.state_dict(),
                "val_loss":        val_loss,
            }, best_path)
            print(f"  Best model updated (val={val_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(encoder, generator, discriminator, opt_g, opt_d, epoch, g_loss, val_loss)

    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
