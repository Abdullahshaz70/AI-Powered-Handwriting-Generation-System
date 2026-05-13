import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """AC-GAN discriminator: real/fake + auxiliary character classifier.
    The classifier head gives the generator an explicit gradient signal
    to produce character-distinguishable outputs, preventing mode collapse."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 512)

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.rf_head  = nn.Linear(512, 1)          # real / fake
        self.cls_head = nn.Linear(512, num_classes) # which character

    def forward(self, image, labels):
        feat      = self.features(image)
        feat      = self.pool(feat).view(feat.size(0), -1)  # (B, 512)
        label_emb = self.label_embed(labels)                 # (B, 512)
        proj      = (feat * label_emb).sum(dim=1, keepdim=True)
        rf_logit  = self.rf_head(feat) + proj                # (B, 1)
        cls_logit = self.cls_head(feat)                      # (B, 62)
        return rf_logit, cls_logit


if __name__ == "__main__":
    d      = Discriminator()
    img    = torch.randn(8, 1, 128, 128)
    labels = torch.randint(0, 62, (8,))
    rf, cls = d(img, labels)
    print("rf:", rf.shape, "cls:", cls.shape)
