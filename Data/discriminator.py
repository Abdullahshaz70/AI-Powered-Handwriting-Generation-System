import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Projection-based conditional discriminator.
    Takes a character image + label, outputs real/fake logit.
    The projection term forces conditioning on character identity."""
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(62, 512)

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
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, 1)

    def forward(self, image, labels):
        feat      = self.features(image)
        feat      = self.pool(feat).view(feat.size(0), -1)   # (B, 512)
        label_emb = self.label_embed(labels)                  # (B, 512)
        proj      = (feat * label_emb).sum(dim=1, keepdim=True)  # (B, 1)
        return self.linear(feat) + proj                       # (B, 1)


if __name__ == "__main__":
    d = Discriminator()
    img    = torch.randn(8, 1, 128, 128)
    labels = torch.randint(0, 62, (8,))
    print(d(img, labels).shape)  # (8, 1)
