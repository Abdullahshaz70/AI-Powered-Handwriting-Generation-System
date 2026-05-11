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
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(__file__))
from skeleton import skeletonize_image, extract_ordered_points
from bezier   import fit_bezier_to_points, curves_to_label, N_CURVES, draw_bezier

CHAR_ORDER  = string.ascii_uppercase + string.ascii_lowercase + string.digits
CHAR_TO_IDX = {c: i for i, c in enumerate(CHAR_ORDER)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHAR_ORDER)}
NUM_CHARS   = len(CHAR_ORDER)   # 62

CANVAS_SIZE = 128
SKIP_DIRS   = {'Writers_Zip', 'output_preview', '__pycache__'}

QUALITY_TOL_PX        = 2.5
MIN_INK_COVERAGE      = 0.65
MAX_RENDER_OUTSIDE    = 0.55
DEFAULT_DROPPED_LOG   = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'dropped_samples.txt'))


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
    if not _passes_quality_filter(arr, curves):
        return None
    return curves_to_label(curves)


def _passes_quality_filter(img_array, curves,
                           tol_px=QUALITY_TOL_PX,
                           min_ink_coverage=MIN_INK_COVERAGE,
                           max_render_outside=MAX_RENDER_OUTSIDE):
    """
    Return True if rendered curves match the original ink reasonably well.
    This is a lenient filter to drop only mislabeled or very poor samples.
    """
    ink_mask = img_array < 128
    if not np.any(ink_mask):
        return False

    canvas = np.ones_like(img_array, dtype=np.uint8) * 255
    rendered = draw_bezier(canvas, curves, thickness=3)
    render_mask = rendered < 128
    if not np.any(render_mask):
        return False

    # Coverage: how much of the original ink is explained by the rendered curves.
    dist_to_render = distance_transform_edt(~render_mask)
    ink_covered = dist_to_render[ink_mask] <= tol_px
    ink_coverage = float(np.mean(ink_covered))

    # Outside: how much of the rendered ink falls away from the original ink.
    dist_to_ink = distance_transform_edt(~ink_mask)
    render_outside = dist_to_ink[render_mask] > tol_px
    outside_ratio = float(np.mean(render_outside))

    return (ink_coverage >= min_ink_coverage) and (outside_ratio <= max_render_outside)


def load_all_data(writers_root, cache_path=None, include_writers=None, dropped_log_path=DEFAULT_DROPPED_LOG):
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

    total_kept = 0
    total_dropped = 0
    dropped_paths = []
    if include_writers is not None:
        include_writers = set(include_writers)

    for entry in sorted(os.scandir(writers_root), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        if include_writers is not None and entry.name not in include_writers:
            continue
        wid = len(writer_names)
        writer_names.append(entry.name)
        n = 0
        dropped = 0
        for fname in sorted(os.listdir(entry.path)):
            if not fname.lower().endswith('.png'):
                continue
            char = _parse_char(fname)
            if char is None:
                continue
            img_path = os.path.join(entry.path, fname)
            if img_path not in label_cache:
                label_cache[img_path] = _extract_label(img_path)
            label = label_cache[img_path]
            if label is None:
                dropped += 1
                dropped_paths.append(img_path)
                continue
            records.append((img_path, CHAR_TO_IDX[char], wid, label))
            n += 1
        total_kept += n
        total_dropped += dropped
        print(f"  {entry.name}: {n} images  (writer_id={wid}, dropped={dropped})")

    if cache_path:
        np.save(cache_path, label_cache)
        print(f'  Saved label cache: {cache_path}')

    if dropped_log_path:
        os.makedirs(os.path.dirname(dropped_log_path), exist_ok=True)
        with open(dropped_log_path, 'w', encoding='utf-8') as f:
            for p in dropped_paths:
                f.write(f'{p}\n')
        print(f'  Saved dropped list: {dropped_log_path}')

    print(f'Total: {len(records)} images, {len(writer_names)} writers, dropped={total_dropped}')
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
