"""
HandwritingCNN — regression network from the blueprint.

Input : grayscale image (1, 128, 128) + character index (int)
Output: 40 floats = 5 Bézier curves × 4 control points × (x, y)

The character embedding lets one model handle all 62 characters
without needing separate per-character models.
"""
import torch
import torch.nn as nn
from bezier import LABEL_DIM, NUM_CHARS

EMBED_DIM = 32


class HandwritingCNN(nn.Module):
    def __init__(self, num_chars=NUM_CHARS, embed_dim=EMBED_DIM, out_dim=LABEL_DIM):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),   # 128→32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),  # 32→8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8→4
            nn.Flatten(),                                                   # 1024
        )
        self.embedding = nn.Embedding(num_chars, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(1024 + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, img, char_idx):
        feat     = self.cnn(img)                       # (B, 1024)
        emb      = self.embedding(char_idx)            # (B, 32)
        combined = torch.cat([feat, emb], dim=1)       # (B, 1056)
        return self.head(combined)                     # (B, 40)


if __name__ == '__main__':
    m = HandwritingCNN()
    x  = torch.randn(4, 1, 128, 128)
    ci = torch.randint(0, 62, (4,))
    out = m(x, ci)
    print(f'Input image : {x.shape}')
    print(f'Char indices: {ci}')
    print(f'Output      : {out.shape}')
    print(f'Params      : {sum(p.numel() for p in m.parameters()):,}')
