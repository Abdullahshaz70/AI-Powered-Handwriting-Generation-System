"""
CharNet — character embedding → Bézier control points.

Input : char_idx  (int, 0–61)
Output: 24 floats = 3 cubic Bézier curves × 4 control points × 2 coords, all in [0,1]

Learns the average stroke shape for each character from real handwriting data.
Natural pen-jitter is added at generation time via add_variation().
"""
import torch
import torch.nn as nn

OUT_DIM = 3 * 4 * 2   # 24


class CharNet(nn.Module):
    def __init__(self, num_chars=62, embed_dim=128):
        super().__init__()

        self.char_emb = nn.Embedding(num_chars, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),                          nn.ReLU(),
            nn.Linear(128, OUT_DIM),
            nn.Sigmoid(),   # output in [0, 1] — matches normalised Bézier coords
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.char_emb.weight, 0.0, 0.1)

    def forward(self, char_idx):
        return self.net(self.char_emb(char_idx))   # (B, 24)


if __name__ == '__main__':
    m  = CharNet(num_chars=62)
    ci = torch.randint(0, 62, (4,))
    y  = m(ci)
    print(f'Output : {y.shape}')   # (4, 24)
    print(f'Range  : [{y.min().item():.3f}, {y.max().item():.3f}]')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
