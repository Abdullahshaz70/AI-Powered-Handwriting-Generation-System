import torch
import torch.nn as nn


class CharacterGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding is now 128-dim (same as style vector) so neither dominates
        self.char_embedding = nn.Embedding(62, 128)
        self.fc = nn.Linear(256, 16384)  # 128 style + 128 char = 256

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, style, labels):
        char_emb = self.char_embedding(labels)
        x = torch.cat([style, char_emb], dim=1)  # (B, 256)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = torch.relu(self.bn1(self.deconv1(x)))
        x = torch.relu(self.bn2(self.deconv2(x)))
        x = torch.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


if __name__ == "__main__":
    generator = CharacterGenerator()
    style = torch.randn(32, 128)
    labels = torch.randint(0, 62, (32,))
    output = generator(style, labels)
    print(output.shape)
