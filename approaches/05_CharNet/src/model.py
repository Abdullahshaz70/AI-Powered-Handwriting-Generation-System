"""
CharNet — character + category embeddings → Bézier control points.

Inputs : char_idx  (0–61)  +  cat_idx (0=uppercase, 1=lowercase, 2=digit)
Output : 48 floats = 6 cubic Bézier curves × 4 pts × 2 coords, all in [0,1]

The category embedding gives the model an explicit group signal so that
uppercase 'A', lowercase 'a', and digit '1' can never collapse to the same curves.
"""
import torch
import torch.nn as nn
from bezier import LABEL_DIM   # 48

NUM_CATEGORIES = 3   # uppercase / lowercase / digit


def char_idx_to_cat(char_idx_tensor):
    """Map char indices to category: 0=upper, 1=lower, 2=digit."""
    cat = torch.zeros_like(char_idx_tensor)
    cat[char_idx_tensor >= 26] = 1   # lowercase a-z
    cat[char_idx_tensor >= 52] = 2   # digits 0-9
    return cat


class CharNet(nn.Module):
    def __init__(self, num_chars=62, embed_dim=128, cat_dim=32):
        super().__init__()

        self.char_emb = nn.Embedding(num_chars,      embed_dim)
        self.cat_emb  = nn.Embedding(NUM_CATEGORIES, cat_dim)

        in_dim = embed_dim + cat_dim   # 160

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    256),                    nn.ReLU(),
            nn.Linear(256, LABEL_DIM),
            nn.Sigmoid(),   # output in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.char_emb.weight, 0.0, 0.1)
        nn.init.normal_(self.cat_emb.weight,  0.0, 0.1)

    def forward(self, char_idx, cat_idx):
        z = torch.cat([self.char_emb(char_idx), self.cat_emb(cat_idx)], dim=1)
        return self.net(z)   # (B, 48)


if __name__ == '__main__':
    m   = CharNet()
    ci  = torch.tensor([0, 26, 52])           # A, a, 0
    cat = char_idx_to_cat(ci)
    y   = m(ci, cat)
    print(f'Output : {y.shape}')              # (3, 48)
    print(f'Range  : [{y.min():.3f}, {y.max():.3f}]')
    print(f'Params : {sum(p.numel() for p in m.parameters()):,}')
