"""
HandwritingCNN — CNN backbone + character embedding → Bézier control points.

Input : reference handwriting image (1, 128, 128) + char_idx
Output: 24 floats = 3 cubic Bézier curves × 4 control points × 2 coords (all in [0,1])

Training: feed real handwriting image of a character → predict its Bézier skeleton curves.
          The CNN learns each writer's stroke style from their real images.
Inference: feed any real image from target writer → get curves for requested character
           in that writer's style.
"""
import torch
import torch.nn as nn

OUT_DIM = 3 * 4 * 2   # N_CURVES * CP_PER_CURVE * 2 = 24


class HandwritingCNN(nn.Module):
    def __init__(self, num_chars=62, embed_dim=32):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(4),   # 128→32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(4),  # 32→8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 8→4
            nn.Flatten(),                                                                        # 64*4*4=1024
        )

        self.char_emb = nn.Embedding(num_chars, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(1024 + embed_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, OUT_DIM),
            nn.Sigmoid(),   # output in [0, 1] — matches normalised Bézier coords
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, char_idx):
        feat = self.cnn(img)                               # (B, 1024)
        emb  = self.char_emb(char_idx)                    # (B, embed_dim)
        return self.head(torch.cat([feat, emb], dim=1))   # (B, 24)


if __name__ == '__main__':
    m   = HandwritingCNN(num_chars=62)
    img = torch.randn(4, 1, 128, 128)
    ci  = torch.randint(0, 62, (4,))
    y   = m(img, ci)
    print(f'Output : {y.shape}')   # (4, 24)
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
