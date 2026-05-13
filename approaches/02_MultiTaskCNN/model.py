import torch
import torch.nn as nn


class MultiTaskCNN(nn.Module):
    def __init__(self, num_writers: int, num_chars: int = 62):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.5)
        self.char_head = nn.Linear(256, num_chars)
        self.writer_head = nn.Linear(256, num_writers)

    def forward(self, x):
        features = self.backbone(x).view(x.size(0), -1)
        features = self.dropout(features)
        return self.char_head(features), self.writer_head(features)

    def extract_features(self, x):
        return self.backbone(x).view(x.size(0), -1)


if __name__ == "__main__":
    model = MultiTaskCNN(num_writers=6)
    x = torch.randn(4, 1, 128, 128)
    char_logits, writer_logits = model(x)
    print(f"Input:          {x.shape}")
    print(f"Char logits:    {char_logits.shape}")
    print(f"Writer logits:  {writer_logits.shape}")
    print(f"Params:         {sum(p.numel() for p in model.parameters()):,}")
