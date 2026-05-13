"""
Dataset: pairs each real handwriting PNG with the font-rendered version
of the same character. The CNN learns to translate font → handwriting.
"""
import os
import string
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from font_render import render_char

# ── character vocabulary ─────────────────────────────────────────────────────
CHAR_ORDER  = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_ORDER)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_ORDER)}
NUM_CHARS   = len(CHAR_ORDER)   # 62

SKIP_DIRS = {'Writers_Zip', 'output_preview', '__pycache__'}


def _parse_char(filename):
    """Extract character from filenames like 'Name_lc_a_r01.png'."""
    parts = os.path.splitext(filename)[0].split('_')
    for i, part in enumerate(parts):
        if part in ('lc', 'uc', 'n') and i + 1 < len(parts):
            candidate = parts[i + 1]
            if len(candidate) == 1 and candidate in CHAR_TO_IDX:
                return candidate
    if len(parts) > 2:
        candidate = parts[2]
        if len(candidate) == 1 and candidate in CHAR_TO_IDX:
            return candidate
    return None


def load_all_data(writers_root):
    """
    Scan Writers_pngs, return (records, writer_names).
    records: list of (img_path, char_idx, writer_idx)
    """
    records      = []
    writer_names = []

    for entry in sorted(os.scandir(writers_root), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        writer_idx = len(writer_names)
        writer_names.append(entry.name)
        n = 0
        for fname in sorted(os.listdir(entry.path)):
            if not fname.lower().endswith('.png'):
                continue
            char = _parse_char(fname)
            if char is None:
                continue
            records.append((os.path.join(entry.path, fname), CHAR_TO_IDX[char], writer_idx))
            n += 1
        print(f"  {entry.name}: {n} images  (writer_id={writer_idx})")

    print(f"Total: {len(records)} images, {len(writer_names)} writers")
    return records, writer_names


class HandwritingDataset(Dataset):
    """
    Returns (font_tensor, real_tensor, char_idx, writer_idx) per sample.

    font_tensor : clean rendered character (CNN input)
    real_tensor : actual handwriting image  (CNN target)
    Both normalised to [-1, 1].
    """

    _base_tf = T.Compose([
        T.Grayscale(1),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    _real_aug = T.Compose([
        T.RandomApply([T.RandomRotation(8)], p=0.5),
        T.RandomApply([T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.92, 1.08))], p=0.4),
    ])

    def __init__(self, records, augment=True):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        real_path, char_idx, writer_idx = self.records[idx]
        char = IDX_TO_CHAR[char_idx]

        # Font image (content) — same character, clean rendering
        font_img = render_char(char)
        font_tensor = self._base_tf(font_img)

        # Real handwriting image (target) — what the CNN should output
        real_img = Image.open(real_path).convert('L')
        if self.augment:
            real_img = self._real_aug(real_img)
        real_tensor = self._base_tf(real_img)

        return font_tensor, real_tensor, char_idx, writer_idx
