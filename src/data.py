"""
Dataset: loads real handwriting PNGs and extracts Bézier labels.
Pipeline per image: PNG → binary → scale_to_canvas → skeletonize → ordered points → Bézier curves → 24 floats
"""
import os
import sys
import string
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from skeleton import skeletonize_image, extract_ordered_points
from bezier   import fit_bezier_to_points, curves_to_label, N_CURVES

CHAR_ORDER  = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_ORDER)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_ORDER)}
NUM_CHARS   = len(CHAR_ORDER)   # 62

CANVAS_SIZE = 128
SKIP_DIRS   = {'Writers_Zip', 'output_preview', '__pycache__'}


def scale_to_canvas(img_array, target_fill=0.80, canvas_size=CANVAS_SIZE):
    """Scale ink region to fill target_fill of canvas, centered. Key fix for tiny output."""
    ys, xs = np.where(img_array < 128)
    if len(xs) == 0:
        return img_array
    x1, x2    = xs.min(), xs.max()
    y1, y2    = ys.min(), ys.max()
    ink        = img_array[y1:y2+1, x1:x2+1]
    h, w       = ink.shape
    target_px  = int(canvas_size * target_fill)
    scale      = min(target_px / max(h, 1), target_px / max(w, 1))
    new_h      = max(1, int(h * scale))
    new_w      = max(1, int(w * scale))
    ink_scaled = np.array(
        Image.fromarray(ink.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    )
    canvas        = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    y_off         = (canvas_size - new_h) // 2
    x_off         = (canvas_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = ink_scaled
    return canvas


def _parse_char(filename):
    parts = os.path.splitext(filename)[0].split('_')
    for i, part in enumerate(parts):
        if part in ('lc', 'uc', 'n') and i + 1 < len(parts):
            c = parts[i + 1]
            if len(c) == 1 and c in CHAR_TO_IDX:
                return c
    if len(parts) > 2:
        c = parts[2]
        if len(c) == 1 and c in CHAR_TO_IDX:
            return c
    return None


def _extract_label(img_path):
    """
    Load image → binary → scale_to_canvas → skeletonize → ordered points → Bézier → 24 floats.
    All coordinates in [0, 1] after canvas scaling.
    """
    arr = np.array(Image.open(img_path).convert('L').resize((CANVAS_SIZE, CANVAS_SIZE)))
    arr = np.where(arr < 128, 0, 255).astype(np.uint8)
    arr = scale_to_canvas(arr, target_fill=0.80)
    skel   = skeletonize_image(arr)
    points = extract_ordered_points(skel)
    curves = fit_bezier_to_points(points, n_curves=N_CURVES)
    return curves_to_label(curves)


def load_all_data(writers_root, cache_path=None):
    """
    Scan Writers_pngs. Returns (records, writer_names).
    records: list of (img_path, char_idx, writer_idx, label_24)
    """
    records, writer_names = [], []

    label_cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            label_cache = np.load(cache_path, allow_pickle=True).item()
            print(f'  Loaded {len(label_cache)} cached labels from {cache_path}')
        except Exception:
            label_cache = {}

    for entry in sorted(os.scandir(writers_root), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        wid = len(writer_names)
        writer_names.append(entry.name)
        n = 0
        for fname in sorted(os.listdir(entry.path)):
            if not fname.lower().endswith('.png'):
                continue
            char = _parse_char(fname)
            if char is None:
                continue
            img_path = os.path.join(entry.path, fname)
            if img_path not in label_cache:
                label_cache[img_path] = _extract_label(img_path)
            records.append((img_path, CHAR_TO_IDX[char], wid, label_cache[img_path]))
            n += 1
        print(f"  {entry.name}: {n} images  (writer_id={wid})")

    if cache_path:
        np.save(cache_path, label_cache)
        print(f'  Saved label cache: {cache_path}')

    print(f'Total: {len(records)} images, {len(writer_names)} writers')
    return records, writer_names


def char_idx_to_cat(char_idx):
    """0=uppercase A-Z, 1=lowercase a-z, 2=digit 0-9."""
    if char_idx < 26:  return 0
    if char_idx < 52:  return 1
    return 2


class BezierDataset(Dataset):
    """
    Lightweight dataset — no image loading during training.
    Returns (char_idx, cat_idx, bezier_label).
    bezier_label: float32, shape (48,), Bézier control points in [0, 1].
    """
    def __init__(self, records):
        self.items = [
            (
                torch.tensor(char_idx, dtype=torch.long),
                torch.tensor(char_idx_to_cat(char_idx), dtype=torch.long),
                torch.from_numpy(label),
            )
            for _, char_idx, _, label in records
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
