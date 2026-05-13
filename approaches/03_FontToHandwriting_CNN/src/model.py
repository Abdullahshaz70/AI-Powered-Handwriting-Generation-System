"""
HandwritingStyleNet — U-Net encoder-decoder.

Input : clean font image (1, 128, 128) + writer_idx (int)
Output: that character in the writer's handwriting style (1, 128, 128)

The writer embedding is injected at the bottleneck so the decoder can
render the correct stroke style while the skip connections preserve
character shape from the encoder.
"""
import torch
import torch.nn as nn


class _Enc(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Dec(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.4))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HandwritingStyleNet(nn.Module):
    """
    Font image + writer ID → handwriting image (same character, writer's style).

    Encoder compresses the font character; the writer embedding is concatenated
    at the 8×8 bottleneck; the decoder upsamples with skip connections so the
    character shape is preserved and only the stroke style changes.
    """

    def __init__(self, num_writers=6, embed_dim=128):
        super().__init__()

        # ── Encoder (128→64→32→16→8) ─────────────────────────────────────
        self.e1 = _Enc(1,   32,  bn=False)   # 128 → 64
        self.e2 = _Enc(32,  64)              # 64  → 32
        self.e3 = _Enc(64,  128)             # 32  → 16
        self.e4 = _Enc(128, 256)             # 16  → 8

        # ── Writer style embedding ────────────────────────────────────────
        self.writer_emb = nn.Embedding(num_writers, embed_dim)

        # ── Decoder (8→16→32→64→128) with skip connections ───────────────
        self.d4 = _Dec(256 + embed_dim, 128, dropout=True)   # 8  → 16
        self.d3 = _Dec(128 + 128,       64,  dropout=True)   # 16 → 32  (+e3 skip)
        self.d2 = _Dec(64  + 64,        32)                  # 32 → 64  (+e2 skip)
        self.d1 = _Dec(32  + 32,        32)                  # 64 → 128 (+e1 skip)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, font_img, writer_idx):
        # Encode
        e1 = self.e1(font_img)    # (B, 32,  64, 64)
        e2 = self.e2(e1)          # (B, 64,  32, 32)
        e3 = self.e3(e2)          # (B, 128, 16, 16)
        e4 = self.e4(e3)          # (B, 256,  8,  8)

        # Inject writer style at bottleneck
        emb = self.writer_emb(writer_idx)                          # (B, 128)
        emb = emb.view(emb.size(0), -1, 1, 1).expand(-1, -1, 8, 8)  # (B, 128, 8, 8)
        x   = torch.cat([e4, emb], dim=1)                         # (B, 384, 8, 8)

        # Decode with skip connections
        x = self.d4(x)                            # (B, 128, 16, 16)
        x = self.d3(torch.cat([x, e3], dim=1))    # (B,  64, 32, 32)
        x = self.d2(torch.cat([x, e2], dim=1))    # (B,  32, 64, 64)
        x = self.d1(torch.cat([x, e1], dim=1))    # (B,  32,128,128)

        return self.out_conv(x)                   # (B,   1,128,128)


if __name__ == '__main__':
    m  = HandwritingStyleNet(num_writers=6)
    x  = torch.randn(4, 1, 128, 128)
    wi = torch.randint(0, 6, (4,))
    y  = m(x, wi)
    print(f'Input : {x.shape}')
    print(f'Output: {y.shape}')
    print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
