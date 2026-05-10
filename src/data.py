"""
Dataset loading: scans Writers_pngs, parses filenames, extracts Bézier labels
via skeletonization, and returns a PyTorch Dataset.

All image processing uses PIL + numpy so cv2 is not required.
"""
import os
import string
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from bezier import LABEL_DIM, fit_bezier_to_points, curves_to_label
from skeleton import skeletonize_image, extract_ordered_points

# ── character vocabulary ─────────────────────────────────────────────────────
CHAR_ORDER  = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_ORDER)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_ORDER)}
NUM_CHARS   = len(CHAR_ORDER)   # 62

SKIP_DIRS = {'Writers_Zip', 'output_preview', '__pycache__'}


# ── filename parser ──────────────────────────────────────────────────────────
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


# ── Bézier label extraction ──────────────────────────────────────────────────
def _binarise(arr):
    """uint8 array: pixels < 128 → black ink (0), else white (255)."""
    return np.where(arr < 128, 0, 255).astype(np.uint8)


def _dilate(arr, radius=2):
    """Simple PIL-based dilation to thicken thin ink strokes."""
    img = Image.fromarray(255 - arr, mode='L')          # invert: ink = white
    from PIL import ImageFilter
    img = img.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return 255 - np.array(img)                          # invert back


def _extract_label(img_path):
    """Load PNG → binarize → dilate → skeletonize → Bézier fit → 40-float label."""
    arr = np.array(Image.open(img_path).convert('L').resize((128, 128)))
    arr = _binarise(arr)
    arr = _dilate(arr, radius=2)
    skeleton = skeletonize_image(arr)
    points   = extract_ordered_points(skeleton)
    curves   = fit_bezier_to_points(points, n_curves=5)
    return curves_to_label(curves)


# ── dataset scanner ──────────────────────────────────────────────────────────
def load_all_data(writers_root, cache_labels=True):
    """
    Scan Writers_pngs, load every PNG, extract Bézier labels.
    Returns (records, writer_names) where records is a list of
    (img_path, char_idx, writer_idx, label_array).

    Labels are cached to Data/bezier_labels.npy on first run.
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

    # ── label cache ───────────────────────────────────────────────────────────
    cache_path = os.path.join(os.path.dirname(writers_root), 'bezier_labels.npy')
    if cache_labels and os.path.exists(cache_path):
        print("Loading cached Bézier labels...")
        cached = np.load(cache_path, allow_pickle=True).item()
        labels = [cached.get(r[0]) for r in records]
        missing = [i for i, l in enumerate(labels) if l is None]
        if missing:
            print(f"  Re-extracting {len(missing)} missing labels...")
            for i in missing:
                labels[i] = _extract_label(records[i][0])
    else:
        print(f"Extracting Bézier labels for {len(records)} images "
              f"(first run — may take a few minutes)...")
        labels = []
        for i, (img_path, _, _) in enumerate(records):
            if i % 200 == 0:
                print(f"  {i}/{len(records)}")
            labels.append(_extract_label(img_path))
        if cache_labels:
            cache = {records[i][0]: labels[i] for i in range(len(records))}
            np.save(cache_path, cache)
            print(f"  Cached to {cache_path}")

    full_records = [
        (records[i][0], records[i][1], records[i][2], labels[i])
        for i in range(len(records))
    ]
    return full_records, writer_names


# ── PyTorch Dataset ──────────────────────────────────────────────────────────
class HandwritingDataset(Dataset):
    _base_tf = T.Compose([
        T.Grayscale(1),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])
    _aug = T.RandomApply([T.RandomRotation(5)], p=0.5)

    def __init__(self, records, augment=True):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, char_idx, _, label = self.records[idx]
        img = Image.open(img_path).convert('L')
        if self.augment:
            img = self._aug(img)
        img = self._base_tf(img)
        return img, char_idx, torch.tensor(label, dtype=torch.float32)
